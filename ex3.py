# ---------------------------- imports ----------------------------------------
import nibabel as nib
from skimage.morphology import binary_closing, binary_erosion, \
    binary_dilation, cube, remove_small_holes, remove_small_objects, \
    binary_opening, flood_fill
from skimage.measure import label, regionprops
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
# ---------------------------Constants ----------------------------------------
DIR_INPUT = 'C:\\Users\\nirma\\PycharmProjects\\Thesis_Medical_Image_Proccesing\\mip_exercises\\Targil1_data\\'
DIR_RESULTS = 'C:\\Users\\nirma\\PycharmProjects\\Thesis_Medical_Image_Proccesing\\mip_exercises\\ex3'
NIFTY_END = ".nii.gz"
CLEAN_BODY = 'CLEAN_BODY'
LUNG_SEG = 'lung seg'

MIN_TH = -500
MAX_TH = 2000
SUCCESS = 1
FAIL = 0
CLEAN_CONSTANT = 5
NUMBER_OF_SEEDS = 200
TH_PRECENT = 3
# ----------------------------Class--------------------------------------------

class Origin_Segmentation(ABC):
    'class that contain API for origin segmentation'

    # ------------Abstract Methods -------------------------------------------
    @abstractmethod
    def find_ROI_segmentation(self):
        """
        Find roi region for the origin the class create segmentation for it
        :return:segmentation the roi
        """
        pass

    @abstractmethod
    def find_origin_segmentation(self, roi_seg, parameters):
       """
       create segmentation for specific roi
       :param roi_seg: the roi of the origin that the class create segmentation
       :param parameters: extra parameters that may help seg the origin
       :return:object segmentation
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





class Liver_Segmentation(Origin_Segmentation):
    'The Liver_Segmentation segment the liver and inherits from the Origin_Segmentation'

    #-------------------------------Constructor -----------------------------

    def __init__(self,ct_scan, aorta_scan, number_of_seeds, num_case, dir_input, dir_results, L1_ct_scan):
        """
        :param th:
        :param ct_scan:
        :param aorta_scan:
        :param number_of_seeds:
        :param num_case:
        """

        self.num_case = num_case

        self.number_of_seeds = number_of_seeds
        self.ct_scan = ct_scan
        self.aorta_scan = aorta_scan
        self.dir_input = dir_input

        val = self.load_nifty_file(ct_scan)
        self.ct_mat = val[0]


        self.ct_file = val[1]

        val = self.load_nifty_file(aorta_scan)
        self.aorta_mat = val[0]
        self.aorta_file = val[1]
        self.dir_results = dir_results

        self.l1_ct_scan = L1_ct_scan
        val = self.load_nifty_file(L1_ct_scan)
        self.l1_mat = val[0]
        self.l1_ct_file = val[1]

    # ----------------------------- Private Methods ---------------------------

    def __choose_largest_co_component(self, seg, num_size):
        """
        :param seg:
        :param num_size:
        :return:
        """

        mask_labels, num = label(seg, return_num = True,
                                 connectivity = 1)
        his, _ = np.histogram(mask_labels, bins = num + 1)
        his = his[1:]
        his_indexs = np.argsort(his)
        print(his.shape[0],"his shape 0")
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
        nib.save(seg_file, self.dir_results +
                 self.ct_scan + '_' + CLEAN_BODY + NIFTY_END)
        return seg


    def __isolate_lung(self, seg):
        """
        :param seg:
        :return:
        """
        cut_seg = np.zeros((seg.shape))
        new_seg1 = np.zeros((seg.shape))
        new_seg1[:, :, :] = 1 - seg[:, :, :]
        new_seg = binary_closing(np.copy(new_seg1), np.ones((3, 3, 3)))
        m_ax = (new_seg.shape[2] * 1) // 2
        cut_seg[:,:,m_ax:] = np.copy(new_seg[:,:,m_ax:])
        cut_seg[:,:,m_ax:] = flood_fill(cut_seg[:,:,m_ax:],(0, 0, 0), 2)
        cut_seg[:, :, m_ax:] = flood_fill(cut_seg[:, :, m_ax:], (
        cut_seg[:, :, m_ax:].shape[0] - 1, cut_seg[:, :, m_ax:].shape[1] - 1, 0), 3)
        new_seg[np.where((cut_seg == 2) | (cut_seg == 3))] = 0
        reg_seg = np.copy(new_seg)
        new_seg[:,:,m_ax:] = self.__choose_largest_co_component(np.copy(reg_seg[:, :, m_ax:]) , 1)
        new_seg = np.array(new_seg, dtype=np.int)
        new_seg[:,:,m_ax:] += self.__choose_largest_co_component(np.copy(reg_seg[:, :, m_ax:]) , 2)
        new_seg[:, :, :m_ax] = 0

        min_sa, min_ca, min_a_a, max_s_a, max_c_a, max_a_a = self.__found_bounding_box(self.aorta_mat)
        new_seg[min_sa: max_s_a,min_ca:max_c_a,min_a_a: max_a_a] = 6
        new_seg = np.array(new_seg, dtype=np.int)
        cut_seg = np.array(cut_seg, dtype=np.int)

        seg_file = nib.Nifti1Image(new_seg, self.ct_file.affine)
        seg_file1 = nib.Nifti1Image(cut_seg, self.ct_file.affine)
        seg_file2 = nib.Nifti1Image(new_seg1, self.ct_file.affine)


        nib.save(seg_file, self.dir_results +
                 self.ct_scan + '_final_lungs_' + LUNG_SEG + NIFTY_END)
        nib.save(seg_file1,  self.dir_results  +
                 self.ct_scan + '_air_and_lung_' + LUNG_SEG + NIFTY_END)
        nib.save(seg_file2, self.dir_results +
                 self.ct_scan + '_raw_' + LUNG_SEG + NIFTY_END)

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
    def __found_bounding_box(self, seg):
        """
        :param seg:
        :return:
        """
        labels = label(seg, return_num = False, connectivity = 2)
        props = regionprops(labels)
        min_s, min_c, min_ax, max_s, max_c, max_ax = props[0]['bbox']
        return min_s, min_c, min_ax, max_s, max_c, max_ax



    def __built_box_roi(self, seg_lung):
        """
        :param seg_lung:
        :return:
        """
        # create box
        seg_liver_check = np.zeros((self.aorta_mat.shape))
        box_seg = np.zeros((self.aorta_mat.shape))
        seg_left_lung = np.zeros((self.aorta_mat.shape))

        # find bounding box for Aorta and lung
        min_sa, min_ca, min_aa, max_sa, max_ca, max_aa = \
            self.__found_bounding_box(self.aorta_mat)
        seg_left_lung[max_sa:, :, :] = seg_lung[max_sa:, :, :]
        min_s_l, min_c_l, min_a_l, max_s_l, max_c_l, max_a_l = \
            self.__found_bounding_box(seg_left_lung)

        min_a_l = np.min(np.where(seg_left_lung != 0)[2])
        max_a_l = np.max(np.where(seg_left_lung != 0)[2])
        # first box
        ss1 = max_s_l - (((max_s_l - min_s_l) * 2) // 3)
        se1 = max_s_l - ((max_s_l - min_s_l) // 5)
        cs1 = min_c_l + ((max_c_l - min_c_l) // 4)
        # ce1 = min_c_l + (((max_c_l - min_c_l) * 2) // 3)
        ce1 = min_c_l + (((max_c_l - min_c_l) * 4) // 5)
        xs1 = min_a_l - ((max_aa - min_aa) // 20)
        xe1 = min_a_l + ((max_aa - min_aa) // 30)

        # second_box
        ss2 = max_s_l - ((max_s_l - min_s_l) // 3)
        se2 = max_s_l - ((max_s_l - min_s_l) // 4)
        cs2 = min_c_l + ((max_c_l - min_c_l) // 4)
        ce2 = min_c_l + (((max_c_l - min_c_l) * 4) // 5)
        # xs2 = min_aa + (((max_aa - min_aa) * 1) // 3)
        xs2 =  xs1 -  (((max_aa - min_aa) * 1) // 5)
        xe2 = xs1

        # third_box
        ss3 = max_sa - ((max_sa - min_sa)//3)
        se3 = max_sa
        cs3 = min_ca
        ce3 = min_ca + (((max_ca - min_ca) * 1)//3)
        xs3 = min_aa + (((max_aa - min_aa)*1) // 2) -  (((max_aa - min_aa)*1) // 40)
        xe3 = min_aa + (((max_aa - min_aa)*1) // 2) +  (((max_aa - min_aa)*1) // 40)


        # update boxes
        box_seg[ss1: se1,cs1:ce1, xs1: xe1] = 1
        box_seg[ss2:  se2,cs2 :ce2, xs2: xe2] = 1
        box_seg[ss3:  se3,cs3 :ce3, xs3: xe3] = 1
        seg_liver_check[min_s_l:max_s_l, min_c_l: max_c_l,min_a_l: max_a_l] = 4
        seg_liver_check[min_sa:max_sa, min_ca: max_ca,min_aa:  max_aa] = 10
        seg_liver_check[np.where(box_seg == 1)] = 1

        # delete soft tissues and hard tissues
        box_seg[np.where((self.ct_mat < -100) | (self.ct_mat > 200))] = 0
        # seg_liver_check[np.where(self.aorta_mat != 0)] = 2

        # add original liver if case equal to one
        if self.num_case == 1 and self.ct_scan != 'Hard' + 'Case' + str(self.num_case) + '_CT.nii.gz':
            val = self.load_nifty_file('Case1_liver_segmentation.nii.gz')
            liver_mat = val[0]
            liver_file = val[1]
            seg_liver_check[np.where(liver_mat == 1)] = 3
            seg_liver_check[np.where(box_seg == 1)] = 1

        return box_seg, seg_liver_check


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
        seg_body = self.__IsolateBody()
        min_s, min_c, _, max_s, max_c, _ = \
            self.__found_bounding_box(seg_body)
        print( min_s, max_s, min_c, max_c, axial_down, axial_up,"borders")
        return min_s, max_s, min_c, max_c, axial_down, axial_up


    def __find_threshold(self, roi_seg):
        """
        Find the threshold for the algorithm
        :return:
        """
        val1 = np.mean(self.ct_mat[np.where(self.l1_mat != 0)])
        print(val1, 'val1')
        val2 = np.mean(self.ct_mat[np.where(roi_seg != 0)])
        print(val2, 'val2')
        th = (np.abs(val2 - val1)*TH_PRECENT) / 100
        print(th, 'th')
        return th


    def __msgr_operations(self, seg):
        """
        clean seg for final results
        :param seg: the segmentation
        :return:
        """

        seg = binary_dilation(seg)
        seg = binary_dilation(seg)
        seg = binary_dilation(seg)
        seg = binary_dilation(seg)
        seg = binary_dilation(seg)
        seg = binary_erosion(seg)
        seg = binary_erosion(seg)
        seg = binary_erosion(seg)
        seg = binary_erosion(seg)
        seg = self.__choose_largest_co_component(seg, 1)
        seg = np.array(seg, dtype=np.int)

        return seg

    # -------------Public Methods----------------------------------------------
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

        file = nib.load(self.dir_input + nifty_file)
        mat = file.get_data()
        return (mat, file)
    # TODO add intersection with clean body
    def find_ROI_segmentation(self):

        seg = self.__IsolateBody()
        seg_lung = self.__isolate_lung(seg)
        box, seg_liver = self.__built_box_roi(seg_lung)
        if self.ct_scan == 'HardCase1_CT.nii.gz':
            box = box[::-1, :, :]
            seg_liver = seg_liver[::-1, :, :]
        seg_file = nib.Nifti1Image(seg_liver, self.ct_file.affine)
        nib.save(seg_file,  self.dir_results +
                 self.ct_scan + '_roi_seg_with_extra' + str(
                     self.num_case) + "_" + NIFTY_END)

        seg_file = nib.Nifti1Image(box, self.ct_file.affine)
        nib.save(seg_file,  self.dir_results  +
                 self.ct_scan + '_roi_seg_only' + str(
            self.num_case) + "_" + NIFTY_END)

        return box, seg_lung


    def find_origin_segmentation(self, roi_seg, parameters):
        seg_lung = parameters[0]
        min_s, max_s, min_c, max_c, axial_min, axial_up = self.__find_borders(seg_lung)
        s_vals, c_vals, a_vals = self.__find_seeds(roi_seg[min_s: max_s, min_c: max_c, axial_min:axial_up])

        shape_mat = self.ct_mat[min_s: max_s, min_c: max_c, axial_min:axial_up].shape
        seg_final = np.zeros((self.ct_mat.shape))
        ct_mat = np.copy(self.ct_mat[min_s: max_s, min_c: max_c, axial_min:axial_up])
        min_val = np.min(ct_mat)
        ct_mat[np.where(self.aorta_mat[min_s: max_s, min_c: max_c, axial_min:axial_up] != 0)] = min_val

        th = self.__find_threshold(roi_seg)
        curr = np.zeros((shape_mat))
        visited = np.zeros((shape_mat))
        discovered = np.zeros((shape_mat))
        visited[s_vals, c_vals, a_vals] = 1
        curr[s_vals, c_vals, a_vals] = 1
        discovered[s_vals, c_vals, a_vals] = 1
        count = 0
        start = datetime.now()
        while (np.any(curr)):
            count = count + 1
            stop = datetime.now()
            if count > 25:
                break
            curr_new = binary_dilation(curr, np.ones((5, 5, 5)))
            curr_new = binary_closing(curr_new, np.ones((5, 5, 5)))
            curr_new = np.array(curr_new, dtype=np.int)
            curr = np.zeros((shape_mat))
            curr_new[np.where(visited != 0)] = 0
            visited[np.where(curr_new != 0)] = 1
            candidates = np.zeros((shape_mat))
            candidates[np.where(curr_new != 0)] = np.copy(
                ct_mat[np.where(curr_new != 0)])
            mean_val = np.mean(ct_mat[np.where(discovered != 0)])
            trash = np.abs(candidates - mean_val)
            candidates[trash >= th] = 0
            candidates[trash < th] = 1
            candidates[np.where(curr_new != 1)] = 0
            s_vals, c_vals, a_vals = np.where(candidates != 0)
            curr[s_vals, c_vals, a_vals] = 1
            discovered[s_vals, c_vals, a_vals] = 2

        seg_final[min_s: max_s, min_c: max_c, axial_min:axial_up][np.where(discovered != 0)] = \
        discovered[np.where(discovered != 0)]
        discovered[np.where(discovered != 0)] = 5
        seg_final[min_s: max_s, min_c: max_c, axial_min:axial_up][np.where(discovered != 0)] = \
        discovered[np.where(discovered != 0)]
        seg_final = self.__msgr_operations(seg_final)
        seg_file = nib.Nifti1Image(seg_final, self.ct_file.affine)
        nib.save(seg_file,
                 self.dir_results + self.ct_scan + '_seg_liver_fff' + NIFTY_END)

        return seg_final



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


def find_roi(num_case):
    liver_obj = Liver_Segmentation( 'Case' + str(num_case) + '_CT.nii.gz',
                                'Case' + str(num_case) + '_Aorta.nii.gz',
                                NUMBER_OF_SEEDS, num_case,DIR_INPUT,DIR_RESULTS + '\\' +  'Case' + str(num_case) + '\\', 'Case' + str(num_case) + '_L1.nii.gz')

    seg, seg_lung = liver_obj.find_ROI_segmentation()

def seg_origin(num_case):
    liver_obj = Liver_Segmentation( 'Case' + str(num_case) + '_CT.nii.gz',
                                   'Case' + str(num_case) + '_Aorta.nii.gz',
                                   NUMBER_OF_SEEDS, num_case ,DIR_INPUT, DIR_RESULTS + '\\'  + 'Case'+str(num_case) + '\\', 'Case' + str(num_case) + '_L1.nii.gz')

    file = nib.load(DIR_RESULTS + '\\' + 'Case'+str(num_case) +'\\'+
                 liver_obj.ct_scan + '_roi_seg_only' + str(
            liver_obj.num_case) + "_" + NIFTY_END)
    seg_roi = file.get_data()

    file = nib.load(DIR_RESULTS + '\\'+ 'Case'+str(liver_obj.num_case) + '\\' +
                 liver_obj.ct_scan + '_final_lungs_' + LUNG_SEG + NIFTY_END)
    seg_lung = file.get_data()

    segf = liver_obj.find_origin_segmentation(seg_roi, [seg_lung])




def origin_mani(num_case):
    liver_obj = Liver_Segmentation('Case' + str(num_case) + '_CT.nii.gz',
                                   'Case' + str(num_case) + '_Aorta.nii.gz',
                                   NUMBER_OF_SEEDS, num_case, DIR_INPUT, DIR_RESULTS + '\\'+ 'Case' + str(num_case) + '\\', 'Case' + str(num_case) + '_L1.nii.gz' )
    file = nib.load(liver_obj.dir_results + liver_obj.ct_scan + '_seg_liver_fff' + NIFTY_END)
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
    # seg = binary_opening(seg)
    seg = choose_largest_co_component(seg, 1)
    seg = np.array(seg, dtype=np.int)
    seg_n = np.zeros((seg.shape))
    seg_file = nib.Nifti1Image(seg, seg_file.affine)
    nib.save(seg_file,DIR_RESULTS + '\\' + 'Case' + str(num_case) + '\\' +
             'Case' + str(
                 num_case) + '_CT.nii.gz' + '_seg_liver_nir' + NIFTY_END)

    if num_case == 1:
        val = liver_obj.load_nifty_file('Case1_liver_segmentation.nii.gz')
        liver_mat = val[0]
        liver_file = val[1]
        print(liver_obj.evaluateSegmentation(liver_mat, seg))
        seg_n[np.where(seg != 0)] = 1
        seg_n[np.where(liver_mat != 0)] = 5
        seg_file = nib.Nifti1Image(seg_n, seg_file.affine)
        nib.save(seg_file,DIR_RESULTS + '\\' +  'Case' + str(num_case) + '\\'+
                 'Case' + str(
                     num_case) + '_CT.nii.gz' + '_seg_liver_nir_l' + NIFTY_END)


# work_with(1)

# def segmentLiver(nifty_file, Aorta_seg_nifty_file, output_name):






if __name__ == '__main__':
    for i in range(1, 2):
        print(i)
        # print(i,"/i")
        # origin_mani(i, 'Hard')
        # seg_origin(i)
        # find_roi(i, 'Hard')
        # seg_origin(i, 'Hard')


