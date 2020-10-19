# ---------------------------- imports ----------------------------------------
import nibabel as nib
from skimage.morphology import binary_closing, binary_erosion, \
    binary_dilation, cube, remove_small_holes, remove_small_objects, \
    binary_opening, flood_fill
from skimage.measure import label, regionprops
import numpy as np
from abc import ABC, abstractmethod

# ---------------------------Constants ----------------------------------------
DIR_INPUT = 'C:\\Users\\nirma\\PycharmProjects\\Thesis_Medical_Image_Proccesing\\mip_exercises\\Targil1_data\\'
NIFTY_END = ".nii.gz"
CLEAN_BODY = 'CLEAN_BODY'
LUNG_SEG = 'lung seg'

MIN_TH = -500
MAX_TH = 2000
SUCCESS = 1
FAIL = 0
CLEAN_CONSTANT = 40
NUMBER_OF_SEEDS = 200
TH = 6


# ----------------------------Code --------------------------------------------


class Origin_Segmentation(ABC):
    'class Father that contain API for origin segmentation'

    # ------------Abstract Methods -------------------------------------------
    @abstractmethod
    def find_ROI_segmentation(self):
        """
        :return:
        """
        pass

    @abstractmethod
    def find_origin_segmentation(self, roi_seg, parameters):
        """
        :return:
        """
        pass

    #  -----------------------------Private Methods --------------------------
    def __calculate_DICE(self, segA, segB):
        """
        Calculate Dice coefficient for two segmentation
        :param segA: First segmentation
        :param segB:Second segmentation
        :return:Dice coefficient
        """
        dice = np.sum(segB[segA == 1]) * 2.0 / (np.sum(segB) + np.sum(segA))
        return dice

    def __calculate_VOD(self, segA, segB):
        """
        Calculate VOD coefficient for two segmentation
        :param segA: First segmentation
        :param segB:Second segmentation
        :return: VOD
        """
        intersection = np.sum(segA * segB)
        seg = np.zeros((segA.shape))
        seg[np.where(segA == 1)] = 1
        seg[np.where(segB == 1)] = 1
        union = np.sum(seg)
        return 1 - (intersection / union)

    # ------------------------Public Methods ----------------------------------
    def evaluateSegmentation(self, GT_seg, est_seg):
        """
        This function is given two segmentations,
        a GT one and an estimated one,
        and returns a tuple of (VOD_result, DICE_result).
        Use the definitions from the lecture slides
        :param GT_seg: gt segmentation
        :param est_seg: the segmentation we evaluate
        :return: DICE coefficient  and VOD.
        """
        GT_seg[np.where(GT_seg != 0)] = 1
        est_seg[np.where(est_seg != 0)] = 1
        d = self.__calculate_DICE(GT_seg, est_seg)
        v = self.__calculate_VOD(GT_seg, est_seg)
        return v, d


    def load_nifty_file(self, nifty_file):
        """
        load nifty file to a matrix
        :param nifty_file: a grayscale NIFTI file
        :return: A matrix and nifty struct
        """

        split_nifty = nifty_file.split('.')
        if len(split_nifty) < 2:
            return FAIL
        print('.' + split_nifty[-2] + '.' + split_nifty[-1])
        if '.' + split_nifty[-2] + '.' + split_nifty[-1] != NIFTY_END:
            return FAIL

        file = nib.load(DIR_INPUT + nifty_file)
        mat = file.get_data()
        return (mat, file)


class Liver_Segmentation(Origin_Segmentation):
    'The Liver_Segmentation segment the liver and inherits from the Origin_Segmentation'

    #  -------------------------------Constructor -----------------------------

    def __init__(self, th, ct_scan, aorta_scan, number_of_seeds, num_case):
        """
        :param th:
        :param ct_scan:
        :param aorta_scan:
        :param number_of_seeds:
        :param num_case:
        """

        self.num_case = num_case
        self.th = th
        self.number_of_seeds = number_of_seeds
        self.ct_scan = ct_scan
        self.aorta_scan = aorta_scan

        val = self.load_nifty_file(ct_scan)
        self.ct_mat = val[0]
        self.ct_file = val[1]

        val = self.load_nifty_file(aorta_scan)
        self.aorta_mat = val[0]
        self.aorta_file = val[1]

    # ----------------------------- Private Methods ---------------------------

    def __choose_largest_co_component(self, seg, num_size):
        """

        :param seg:
        :param num_size:
        :return:
        """
        mask_labels, num = label(seg, return_num=True,
                                 connectivity=1)
        his, _ = np.histogram(mask_labels, bins=num + 1)
        his = his[1:]
        his_indexs = np.argsort(his)
        index = his.shape[0] - num_size
        mask_labels[np.where(mask_labels != his_indexs[index] + 1)] = 0
        seg[np.where(mask_labels == 0)] = 0
        seg = np.array(seg, dtype=np.int)
        return seg



    def __activate_threshold(self, mat, Imin, Imax):
        """
        The function applies thresholding on a three-dimensional matrix.
        :param mat: the Matrix to threshold
         :param Imin:  minimal threshold
        :param Imax:maximal threshold
        :return: matrix after thresholding
        """
        seg = np.zeros((mat.shape))
        seg[np.where((mat >= Imin) & (mat <= Imax))] = 1
        return seg

    def __clean_body(self, seg):
        """
        :param seg:
        :return:
        """
        seg = binary_opening(seg)
        seg = remove_small_objects(seg, CLEAN_CONSTANT, connectivity=2)
        seg = remove_small_holes(seg)
        seg = binary_dilation(seg, cube(3))
        seg = remove_small_holes(seg)
        return seg

    def __IsolateBody(self):
        """
        :return:
        """
        seg = self.__activate_threshold(self.ct_mat, MIN_TH, MAX_TH)
        seg = self.__clean_body(seg)

        # delete components until there is only one!
        seg = self.__choose_largest_co_component(seg, 1)

        # save nifty file
        seg_file = nib.Nifti1Image(seg, self.ct_file.affine)
        nib.save(seg_file,
                 self.ct_scan + '__' + CLEAN_BODY + NIFTY_END)

        return seg


    def __isolate_lung(self, seg):
        """
        :param seg:
        :return:
        """
        cut_seg = np.zeros((seg.shape))
        new_seg = np.zeros((seg.shape))
        new_seg[:, :, :] = 1 - seg[:, :, :]
        m_ax = (new_seg.shape[2]*4) // 7
        cut_seg[:,:,m_ax:] = np.copy(new_seg[:,:,m_ax:])
        cut_seg[:,:,m_ax:] = flood_fill(cut_seg[:,:,m_ax:],(0,0,0), 2)
        cut_seg[:, :, m_ax:] = flood_fill(cut_seg[:, :, m_ax:], (
        cut_seg[:, :, m_ax:].shape[0] - 1, cut_seg[:, :, m_ax:].shape[1] - 1,
        0), 3)
        new_seg[np.where((cut_seg == 2) | (cut_seg == 3))] = 0
        new_seg[:,:,m_ax:] = self.__choose_largest_co_component(np.copy(new_seg[:,:,m_ax:]) , 1)
        new_seg[:, :, :m_ax] = 0
        new_seg = np.array(new_seg, dtype=np.int)
        cut_seg = np.array(cut_seg, dtype=np.int)
        seg_file = nib.Nifti1Image(new_seg, self.ct_file.affine)
        seg_file1 = nib.Nifti1Image(cut_seg, self.ct_file.affine)


        nib.save(seg_file,
                 self.ct_scan + '_lung_' + LUNG_SEG + NIFTY_END)
        nib.save(seg_file1,
                 self.ct_scan + '_cut seg_' + LUNG_SEG + NIFTY_END)

        return new_seg

    # def built_box_roi(self, seg_lung, liver):
    #     seg_liver_check = np.zeros((self.aorta_mat.shape))
    #     box_seg = np.zeros((self.aorta_mat.shape))
    #     seg_left_lung = np.zeros((self.aorta_mat.shape))
    #
    #     # find bounding box for Aorta and lung
    #     labels = label(self.aorta_mat, return_num=False, connectivity=2)
    #     props = regionprops(labels)
    #     print(props[0]['bbox'])
    #     min_s_a, min_c_a, min_a_a , max_s_a, max_c_a, max_a_a = props[0]['bbox']
    #     s_a_delta, c_a_delta ,a_a_delta = max_s_a - min_s_a,max_c_a - min_c_a, max_a_a - min_a_a
    #     seg_left_lung[max_s_a:, :, :] = seg_lung[max_s_a:, :, :]
    #     labels = label(seg_left_lung, return_num=False, connectivity=2)
    #     props = regionprops(labels)
    #     print(props[0]['bbox'])
    #     min_s_l, min_c_l, min_a_l, max_s_l, max_c_l, max_a_l = props[0]['bbox']
    #     s_box = min_s_l + ((max_s_l - min_s_l)*2) // 3
    #     c_box = min_c_l + (max_c_l - min_c_l) // 2
    #     a_min = min_a_a + (((max_a_a - min_a_a) * 2) // 8)
    #     a_max = np.minimum(min_a_a + (((max_a_a - min_a_a) * 5) // 8), min_a_l)
    #
    #     box_seg[s_box - (s_a_delta // 2):s_box + (s_a_delta // 2),c_box - (c_a_delta // 4):c_box + (c_a_delta // 4), a_min:a_max] = 1
    #     box_seg[np.where((self.ct_mat < -100) | (self.ct_mat > 200))] = 0
    #     print( s_box - (s_a_delta // 2), s_box + (s_a_delta // 2), c_box - (c_a_delta // 4), c_box + (c_a_delta // 4), a_min, a_max)
    #     # seg_liver_check[:, :, :] = aorta_mat[:, :, :]
    #     seg_liver_check[np.where(box_seg == 1)] = 1
    #     seg_liver_check[np.where(self.aorta_mat != 0)] = 2
    #     if self.num_case == 1:
    #         seg_liver_check[np.where(liver == 1)] = 3
    #         seg_liver_check[np.where(box_seg == 1)] = 1
    #
    #     seg_liver_check[min_s_l:max_s_l ,min_c_l: max_c_l, min_a_l: max_a_l] = 4
    #     # seg_liver_check[np.where(liver == 1)] = 3
    #
    #     # seg_liver_check[np.where(seg_lung == 1)] = seg_lung[np.where(seg_lung == 1)]
    #     return box_seg, seg_liver_check, max_s_a
    #
    def __built_box_roi(self, seg_lung, liver):
        """
        :param seg_lung:
        :param liver:
        :return:
        """
        # create box
        seg_liver_check = np.zeros((self.aorta_mat.shape))
        box_seg = np.zeros((self.aorta_mat.shape))
        seg_left_lung = np.zeros((self.aorta_mat.shape))

        # find bounding box for Aorta and lung
        labels = label(self.aorta_mat, return_num=False, connectivity=2)
        props = regionprops(labels)
        print(props[0]['bbox'])
        min_s_a, min_c_a, min_a_a, max_s_a, max_c_a, max_a_a = props[0]['bbox']
        s_a_delta, c_a_delta, a_a_delta = max_s_a - min_s_a, max_c_a - min_c_a, max_a_a - min_a_a
        seg_left_lung[max_s_a:, :, :] = seg_lung[max_s_a:, :, :]
        labels = label(seg_left_lung, return_num=False, connectivity=2)
        props = regionprops(labels)
        print(props[0]['bbox'])
        min_s_l, min_c_l, min_a_l, max_s_l, max_c_l, max_a_l = props[0]['bbox']

        # first box
        s_box1_s = max_s_l - (((max_s_l - min_s_l) * 2) // 3)
        s_box1_e = max_s_l - ((max_s_l - min_s_l) // 5)
        c_box1_s = min_c_l + ((max_c_l - min_c_l) // 4)
        c_box1_e = min_c_l + (((max_c_l - min_c_l) * 2) // 3)
        x_box1_s = min_a_l - ((max_a_a - min_a_a) // 20)
        x_box1_e = min_a_l + ((max_a_a - min_a_a) // 30)

        # second_box
        s_box2_s = max_s_l - ((max_s_l - min_s_l) // 3)
        s_box2_e = max_s_l - ((max_s_l - min_s_l) // 4)
        c_box2_s = min_c_l + ((max_c_l - min_c_l) // 4)
        c_box2_e = min_c_l + (((max_c_l - min_c_l) * 2) // 3)
        x_box2_s = min_a_a + (((max_a_a - min_a_a) * 1) // 3)
        x_box2_e = min_a_l - ((max_a_a - min_a_a) // 30)

        # update boxes
        box_seg[s_box1_s: s_box1_e,
        c_box1_s:c_box1_e, x_box1_s: x_box1_e] = 1
        box_seg[s_box2_s: s_box2_e,
        c_box2_s:c_box2_e, x_box2_s: x_box2_e] = 1
        seg_liver_check[min_s_l:max_s_l, min_c_l: max_c_l,
        min_a_l: max_a_l] = 4
        seg_liver_check[np.where(box_seg == 1)] = 1

        # delete soft tissues and hard tissues
        box_seg[np.where((self.ct_mat < -100) | (self.ct_mat > 200))] = 0

        seg_liver_check[np.where(self.aorta_mat != 0)] = 2
        if self.num_case == 1:
            seg_liver_check[np.where(liver == 1)] = 3
            seg_liver_check[np.where(box_seg == 1)] = 1

        return box_seg, seg_liver_check, max_s_a

    # TODO add number of seeds as class parameter
    def __find_seeds(self, roi_seg):
        """
        :param roi_seg:
        :return:
        """
        sagittal_vals_ao, coronal_vals_ao, axials_vals_ao = np.where(
            roi_seg != 0)
        indexes = np.arange(0, sagittal_vals_ao.shape[0])
        np.random.shuffle(indexes)
        indexes = indexes[:self.number_of_seeds]
        sagittal_vals, coronal_vals, axial_vals = \
            sagittal_vals_ao[indexes], coronal_vals_ao[indexes], \
            axials_vals_ao[indexes]

        return sagittal_vals, coronal_vals, axial_vals

    def __find_borders(self, seg_lung):
        """
        :param seg_lung:
        :return:
        """
        sagittal_vals, coronal_vals, axials_vals = np.where(
            seg_lung != 0)
        axial_up = np.int(np.mean(axials_vals))
        sagittal_vals, coronal_vals, axials_vals = np.where(
            self.aorta_mat != 0)
        axial_down = np.int(np.min(axials_vals))

        return axial_down, axial_up

    # -------------Public Methods----------------------------------------------
    # TODO add intersection with clean body
    def find_ROI_segmentation(self):

        val = self.load_nifty_file('Case1_liver_segmentation.nii.gz')
        if val == 0:
            return FAIL
        liver_mat = val[0]
        liver_file = val[1]

        seg = self.__IsolateBody()
        seg_lung = self.__isolate_lung(seg)
        # box = self.built_roi_for_liver(aorta_mat, ct_mat, seg_lung)
        box, seg_liver, max_s_a = self.__built_box_roi(seg_lung, liver_mat)
        seg_file = nib.Nifti1Image(seg_liver, self.ct_file.affine)
        nib.save(seg_file,
                 self.ct_scan + '_liverbox_' + str(
                     self.num_case) + "__" + NIFTY_END)

        return box, seg_lung, max_s_a





    def find_origin_segmentation(self, roi_seg, parameters):
        seg_lung = parameters[0]
        max_s_a = parameters[1]
        axial_min, axial_up = self.__find_borders(seg_lung)
        print(roi_seg[max_s_a:, :, axial_min:axial_up].shape, "livr roi seg")
        s_vals, c_vals, a_vals = self.__find_seeds(
            roi_seg[max_s_a:, :, axial_min:axial_up])

        val = self.load_nifty_file('Case1_liver_segmentation.nii.gz')
        if val == 0:
            return FAIL
        liver_mat = val[0]
        liver_file = val[1]

        shape_mat = self.ct_mat[max_s_a:, :, axial_min:axial_up].shape
        seg_final = np.zeros((self.ct_mat.shape))
        seg_final2 = np.zeros((self.ct_mat.shape))
        ct_mat = np.copy(self.ct_mat[max_s_a:, :, axial_min:axial_up])
        min_val = np.min(ct_mat)
        ct_mat[np.where(
            self.aorta_mat[max_s_a:, :, axial_min:axial_up] != 0)] = min_val
        curr = np.zeros((shape_mat))
        visited = np.zeros((shape_mat))
        discovered = np.zeros((shape_mat))
        visited[s_vals, c_vals, a_vals] = 1
        curr[s_vals, c_vals, a_vals] = 1
        discovered[s_vals, c_vals, a_vals] = 1
        count = 0
        while (np.any(curr)):
            count = count + 1
            if count > 25:
                break
            print(count)
            print(np.sum(discovered))
            curr_new = binary_dilation(curr, np.ones((5, 5, 5)))
            # curr_new = binary_dilation(curr_new, np.ones((5,5,5)))
            # curr_new = binary_dilation(curr_new)
            curr_new = binary_closing(curr_new, np.ones((5, 5, 5)))
            curr_new = np.array(curr_new, dtype=np.int)
            curr = np.zeros((shape_mat))
            curr_new[np.where(visited != 0)] = 0
            visited[np.where(curr_new != 0)] = 1
            # ---------------------------------------
            candidates = np.zeros((shape_mat))
            candidates[np.where(curr_new != 0)] = np.copy(
                ct_mat[np.where(curr_new != 0)])
            mean_val = np.mean(ct_mat[np.where(discovered != 0)])
            print(np.sum(discovered) // 2, "discoverd")
            print(mean_val, "mean val")
            trash = np.abs(candidates - mean_val)
            candidates[trash >= TH] = 0
            candidates[trash < TH] = 1
            candidates[np.where(curr_new != 1)] = 0
            s_vals, c_vals, a_vals = np.where(candidates != 0)
            curr[s_vals, c_vals, a_vals] = 1
            discovered[s_vals, c_vals, a_vals] = 2

        seg_final[max_s_a:, :, axial_min:axial_up][np.where(discovered != 0)] = \
        discovered[np.where(discovered != 0)]
        # seg_final[np.where(liver_mat != 0)] = liver_mat[
        #     np.where(liver_mat != 0)]
        discovered[np.where(discovered != 0)] = 5
        seg_final[max_s_a:, :, axial_min:axial_up][np.where(discovered != 0)] = \
        discovered[np.where(discovered != 0)]

        seg_file = nib.Nifti1Image(seg_final, self.ct_file.affine)
        nib.save(seg_file,
                 self.ct_scan + '_seg_liver_fff' + NIFTY_END)

        return seg_final


# # seg_liver_check[:, :, start:] = aorta_mat[:, :, start:]
# seg_liver_check[np.where(box_seg == 2)] = box_seg[
#     np.where(box_seg == 2)]
# # seg_liver_check[np.where(seg_lung == 1)] = seg_lung[np.where(seg_lung == 1)]
#
#
# seg_file = nib.Nifti1Image(seg_liver_check, ct_file.affine)
# nib.save(seg_file,
#          ct_scan + '_liver_' + NIFTY_END)
#
# seg_liver_check[np.where(liver_mat != 0)] = liver_mat[np.where(liver_mat != 0)]
# seg_file = nib.Nifti1Image(seg_liver_check, ct_file.affine)
# nib.save(seg_file,
#          ct_scan + '_liver_with_liver' + NIFTY_END)
# -------------------------------- End class ----------------------------------

def choose_largest_co_component(seg, num_size):
    """

    :param seg:
    :param num_size:
    :return:
    """
    mask_labels, num = label(seg, return_num=True,
                             connectivity=1)
    his, _ = np.histogram(mask_labels, bins=num + 1)
    his = his[1:]
    his_indexs = np.argsort(his)
    index = his.shape[0] - num_size
    mask_labels[np.where(mask_labels != his_indexs[index] + 1)] = 0
    seg[np.where(mask_labels == 0)] = 0
    seg = np.array(seg, dtype=np.int)
    return seg


def check_all(num_case):
    liver_obj = Liver_Segmentation(TH, 'Case' + str(num_case) + '_CT.nii.gz',
                                'Case' + str(num_case) + '_Aorta.nii.gz',
                                NUMBER_OF_SEEDS, num_case)

    seg, seg_lung, max_s_a = liver_obj.find_ROI_segmentation()
    seg1 = liver_obj.find_origin_segmentation(seg, [seg_lung, max_s_a])
    if num_case == 1:
        val = liver_obj.load_nifty_file('Case1_liver_segmentation.nii.gz')
        liver_mat = val[0]
        liver_file = val[1]



def work_with(num_case):
    liver_obj = Liver_Segmentation(TH, 'Case' + str(num_case) + '_CT.nii.gz',
                                   'Case' + str(num_case) + '_Aorta.nii.gz',
                                   NUMBER_OF_SEEDS, num_case)
    file = nib.load(
        'Case' + str(num_case) + '_CT.nii.gz' + '_seg_liver_fff.nii.gz')
    mat = file.get_data()
    val = (mat, file)
    seg = val[0]
    seg_file = val[1]
    seg = binary_dilation(seg)
    seg = binary_dilation(seg)
    seg = binary_dilation(seg)
    seg = binary_dilation(seg)
    seg = binary_dilation(seg)
    seg = binary_erosion(seg)
    seg = binary_erosion(seg)
    seg = binary_erosion(seg)
    seg = binary_erosion(seg)
    seg = choose_largest_co_component(seg, 1)
    # seg = binary_erosion(seg)
    # seg = binary_erosion(seg)

    seg = np.array(seg, dtype=np.int)
    seg_n = np.zeros((seg.shape))
    seg_file = nib.Nifti1Image(seg, seg_file.affine)
    nib.save(seg_file,
             'Case' + str(
                 num_case) + '_CT.nii.gz' + '_seg_liver_nir' + NIFTY_END)
    if num_case == 1:
        val = liver_obj.load_nifty_file('Case1_liver_segmentation.nii.gz')
        liver_mat = val[0]
        liver_file = val[1]
        print(liver_obj.evaluateSegmentation(liver_mat, seg))
        seg_n[np.where(seg != 0)] = 1
        seg_n[np.where(liver_mat != 0)] = 5
        seg_file = nib.Nifti1Image(seg, seg_file.affine)
        nib.save(seg_file,
                 'Case' + str(
                     num_case) + '_CT.nii.gz' + '_seg_liver_nir_l' + NIFTY_END)


# work_with(1)

if __name__ == '__main__':
    for i in range(1, 6):
        print(i,"i")
        # check_all(i)
        work_with(i)


# arr = np.zeros((10,10))
# arr[5,5] = 1
# print(arr)
# print('\n')
# arr = binary_dilation(arr)
#
# print(np.array(arr, dtype=np.int))

#
# check_all(1)

# a = np.zeros((10,10))
#
# a[:,0:4] = 1
# print(a)
# print("\n")
# b = white_tophat(a)
# print(b)


# def built_roi_for_liver(self, aorta_mat, ct_mat, seg_lung):
#
#     # init segs
#     box_seg = np.zeros((aorta_mat.shape))
#     aorta_half = np.zeros((aorta_mat.shape))
#     # borders of axis values
#     sagittal_vals_ao, coronal_vals_ao, axials_vals_ao = np.where(
#         aorta_mat != 0)
#     sagittal_vals_l, coronal_vals_l, axials_vals_l = np.where(
#         seg_lung != 0)
#     diff_axial_ao = np.max(axials_vals_ao) - np.min(axials_vals_ao)
#     start_axial = np.min(axials_vals_ao) + ((diff_axial_ao * 3) // 7)
#     end_axial = np.minimum(np.min(axials_vals_ao) + ((diff_axial_ao * 5) // 8), np.min(axials_vals_l ))
#     # border of saggital and coronal
#     start = np.min(axials_vals_ao) + (diff_axial_ao // 2)
#     aorta_half[:, :, start:] = aorta_mat[:, :, start:]
#     sagittal_vals, coronal_vals, axials_vals = np.where(aorta_half != 0)
#     sagg_index = np.argmin(axials_vals)
#     sagg_value = sagittal_vals[sagg_index]
#     seg_lung[:sagg_value, :, :] = 0
#     sagittal_vals, coronal_vals, axials_vals = np.where(seg_lung != 0)
#     r_sa = (np.max(sagittal_vals) - np.min(sagittal_vals)) // 4
#     r_co = (np.max(coronal_vals) - np.min(coronal_vals)) // 5
#     m_s = np.min(sagittal_vals) + (
#                 (np.max(sagittal_vals) - np.min(sagittal_vals)) * 2) // 3
#     m_c = np.min(coronal_vals) + (
#                 (np.max(coronal_vals) - np.min(coronal_vals)) * 2) // 3
#     # update box and remove un wanted tissues
#     box_seg[m_s - r_sa: m_s + r_sa, m_c - r_co:m_c + r_co,
#     start_axial:end_axial] = 2
#     box_seg[np.where((ct_mat < -100) | (ct_mat > 200))] = 0
#     return box_seg


# def find_ROI(self, return_np_array=False):
#     lungs_seg = self.isolate_lungs(return_np_array=True)
#     min_z_lungs = np.min(np.where(lungs_seg == 1)[2])
#     img = nib.load(self.input_dir + '/' + self.ROI_file + '.nii.gz')
#     l1_seg = img.get_data()
#     labels, num_components = label(l1_seg, return_num=True, connectivity=3)
#     props = regionprops(labels)
#     aorta_bbox = props[0]['bbox']
#     mask = np.zeros_like(l1_seg).astype(np.int)
#     x_delta, y_delta, z_delta = aorta_bbox[3] - aorta_bbox[0], aorta_bbox[
#         4] - aorta_bbox[1], aorta_bbox[5] - aorta_bbox[2]
#     min_x, max_x = aorta_bbox[3] + x_delta * 0.8, aorta_bbox[
#         3] + x_delta * 1.8
#     min_y, max_y = aorta_bbox[1] + y_delta * 0.25, aorta_bbox[
#         4] + y_delta * 0.25
#     min_z, max_z = aorta_bbox[2] + z_delta * (3 / 8), min(
#         aorta_bbox[2] + z_delta * (5 / 8), min_z_lungs)
#
#     mask[int(min_x):int(max_x), int(min_y):int(max_y),
#     int(min_z):int(max_z)] = 1
#
#     ct_img = nib.load(
#         self.input_dir + '/' + self.nifti_file + '.nii.gz').get_data()
#     body_seg = nib.load(
#         OUTPUT + self.nifti_file + '_Body_seg' + '.nii.gz').get_data()
#
#     mask[np.where((ct_img >= 200) | (ct_img <= -100))] = 0
#     mask[np.where(body_seg == 0)] = 0
#     if return_np_array: return mask
#     ref = img.affine
#     nifti_mask = nib.Nifti1Image(mask, ref)
#     nib.save(nifti_mask,
#              OUTPUT + self.nifti_file + '_ROI_Liver' + '.nii.gz')
