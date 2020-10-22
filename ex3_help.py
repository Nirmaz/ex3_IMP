# ---------------------------- imports ----------------------------------------
import nibabel as nib
from skimage.morphology import binary_closing, binary_erosion, \
    binary_dilation, remove_small_objects, \
    binary_opening, flood_fill
from skimage.measure import label, regionprops
import numpy as np
from abc import ABC, abstractmethod
# ---------------------------Constants ----------------------------------------
DIR_INPUT = 'C:\\Users\\nirma\\PycharmProjects\\Thesis_Medical_Image_Proccesing\\mip_exercises\\Targil1_data\\'
DIR_RESULTS = 'C:\\Users\\nirma\\PycharmProjects\\Thesis_Medical_Image_Proccesing\\mip_exercises\\ex3'
NIFTY_END = ".nii.gz"
CLEAN_BODY = 'CLEAN_BODY'


MIN_TH = -500
MAX_TH = 2000
SUCCESS = 1
FAIL = 0
CLEAN_CONSTANT = 5
NUMBER_OF_SEEDS = 200
TH_PRE = 3
NUMBER_OF_IT = 25
AXIAL_CON = 10
S_CON = 3
SOFT_TISSUE = -100
BONES = 200
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

    @abstractmethod
    def find_origin_all(self):
        """
        Find roi region for the origin and afterwords finds segmentation
        for the origin and save it in the results path
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

    # -------------------------------Constructor -----------------------------

    def __init__(self, nifty_file, aorta_nifty_file,
                 dir_input, dir_results, name_results, L1_ct_scan):
        """
        Constructor for init segmentation
        :param nifty_file: the file that contain the CT segmentation
        :param aorta_nifty_file: the aorta file that contain the CT segmentation
        :param dir_input: the dir that the nifty file
        :param dir_results: the dir to save the results
        :param name_results: the name fo the live seg results file
        """

        self.ct_scan = nifty_file
        self.aorta_scan = aorta_nifty_file
        self.dir_input = dir_input

        # load results
        val = self.load_nifty_file(nifty_file)
        self.ct_mat = val[0]
        self.ct_file = val[1]

        # load aorta
        val = self.load_nifty_file(aorta_nifty_file)
        self.aorta_mat = val[0]
        self.aorta_file = val[1]

        # results path
        self.dir_results = dir_results
        self.name__results = name_results

        # l1 scan
        self.l1_ct_scan = L1_ct_scan
        val = self.load_nifty_file(L1_ct_scan)
        self.l1_mat = val[0]
        self.l1_ct_file = val[1]
    # ----------------------------- Private Methods ---------------------------

    def __choose_largest_co_component(self, seg, num_size):
        """
       Delete all the connectivity components except the component, that
       if we will arrange all the connectivity components according to their
       size. The components will be rank as num size.
       :param seg: A segmentation matrix
       :param num_size: the rank of the chosen matrix
       :return:the segmentation matrix with the largest connectivity component
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


    def __IsolateBody(self):
        """
        create segmentation of only the human body
        :return:segmentation of the human body
        """
        seg = self.__activate_threshold(self.ct_mat, MIN_TH, MAX_TH)

        # clean noise
        seg = binary_opening(seg)
        seg = remove_small_objects(seg, CLEAN_CONSTANT, connectivity=2)

        # delete components until there is only one!
        seg = self.__choose_largest_co_component(seg, 1)

        # save nifty file
        seg_file = nib.Nifti1Image(seg, self.ct_file.affine)
        nib.save(seg_file, self.dir_results +
                 self.ct_scan + '_' + CLEAN_BODY+'a' + NIFTY_END)
        return seg

    def __isolate_lung(self, seg):
        """
        Segment the lung in the body
        :param seg: the segmentation of the human body
        :return: segmentation of lung
        """
        # take only the parts that contain air
        cut_seg = np.zeros((seg.shape))
        new_seg1 = np.zeros((seg.shape))
        new_seg1[:, :, :] = 1 - seg[:, :, :]
        new_seg = binary_closing(np.copy(new_seg1), np.ones((3, 3, 3)))

        # delete air that is outside the body
        m_ax = (new_seg.shape[2] * 1) // 2
        cut_seg[:, :, m_ax:] = np.copy(new_seg[:, :, m_ax:])
        cut_seg[:, :, m_ax:] = flood_fill(cut_seg[:, :, m_ax:], (0, 0, 0), 2)
        cut_seg[:, :, m_ax:] = flood_fill(cut_seg[:, :, m_ax:], (
            cut_seg[:, :, m_ax:].shape[0] - 1,
            cut_seg[:, :, m_ax:].shape[1] - 1, 0), 3)
        new_seg[np.where((cut_seg == 2) | (cut_seg == 3))] = 0
        reg_seg = np.copy(new_seg)
        new_seg[:, :, m_ax:] = self.__choose_largest_co_component(
            np.copy(reg_seg[:, :, m_ax:]), 1)
        new_seg[:, :, :m_ax] = 0

        return new_seg


    def __found_bounding_box(self, seg):
        """
        Create bounding box around segmentation object
        :param seg: segmentation matrix
        :return: the borders of the segmentation matrix
        """
        labels = label(seg, return_num=False, connectivity=2)
        props = regionprops(labels)
        min_s, min_c, min_ax, max_s, max_c, max_ax = props[0]['bbox']
        return min_s, min_c, min_ax, max_s, max_c, max_ax

    def __built_box_roi(self, seg_lung):
        """
        Create a roi for the liver
        :param seg_lung: seg of man lungs
        :return: seg of the roi of the liver
        """
        # create box and roi of left lung
        box_seg = np.zeros((self.aorta_mat.shape))
        seg_left_lung = np.zeros((self.aorta_mat.shape))

        # find bounding box for Aorta and lung
        min_sa, min_ca, min_aa, max_sa, max_ca, max_aa = \
            self.__found_bounding_box(self.aorta_mat)
        seg_left_lung[max_sa:, :, :] = seg_lung[max_sa:, :, :]
        min_s_l, min_c_l, min_a_l, max_s_l, max_c_l, max_a_l = \
            self.__found_bounding_box(seg_left_lung)

        # first box
        ss1 = max_s_l - (((max_s_l - min_s_l) * (S_CON-1)) // S_CON)
        se1 = max_s_l - ((max_s_l - min_s_l) // (S_CON+2))
        cs1 = min_c_l + ((max_c_l - min_c_l) // (S_CON+1))
        ce1 = min_c_l + (((max_c_l - min_c_l) * (S_CON+1)) // (S_CON+2))
        xs1 = min_a_l - ((max_aa - min_aa) // AXIAL_CON*2)
        xe1 = min_a_l + ((max_aa - min_aa) // AXIAL_CON*3)

        # second_box
        ss2 = max_s_l - ((max_s_l - min_s_l) // S_CON)
        se2 = max_s_l - ((max_s_l - min_s_l) // (S_CON+1))
        cs2 = min_c_l + ((max_c_l - min_c_l) // (S_CON+1))
        ce2 = min_c_l + (((max_c_l - min_c_l) * (S_CON+1)) // (S_CON+2))
        xs2 = min_aa + (((max_aa - min_aa) * 1) // (S_CON+1))
        xe2 = min_a_l - ((max_aa - min_aa) // AXIAL_CON*3)

        # third_box
        ss3 = max_sa - ((max_sa - min_sa) // S_CON)
        se3 = max_sa
        cs3 = min_ca
        ce3 = min_ca + (((max_ca - min_ca) * 1) // S_CON)
        xs3 = min_aa + (((max_aa - min_aa) * ((AXIAL_CON//2) - 1) ) // (AXIAL_CON - 1))
        xe3 = min_aa + (((max_aa - min_aa) * (AXIAL_CON//2)) // (AXIAL_CON - 1))

        # update boxes
        box_seg[ss1: se1, cs1:ce1, xs1: xe1] = 1
        box_seg[ss2:  se2, cs2:ce2, xs2: xe2] = 1
        box_seg[ss3:  se3, cs3:ce3, xs3: xe3] = 1

        # delete soft tissues and hard tissues
        box_seg[np.where((self.ct_mat < SOFT_TISSUE) | (self.ct_mat > BONES))] = 0

        return box_seg

    def __find_seeds(self, roi_seg):
        """
        sample seeds from from the roi region
        :param roi_seg: the roi segmentation
        :return: the locations of the seeds
        """
        sagittal_vals_ao, coronal_vals_ao, axials_vals_ao = np.where(
            roi_seg != 0)
        indexes = np.arange(0, sagittal_vals_ao.shape[0])
        np.random.shuffle(indexes)
        indexes = indexes[:NUMBER_OF_SEEDS]
        sagittal_vals, coronal_vals, axial_vals = \
            sagittal_vals_ao[indexes], coronal_vals_ao[indexes], \
            axials_vals_ao[indexes]

        return sagittal_vals, coronal_vals, axial_vals

    def __find_borders(self, seg_lung):
        """
        find the border of the liver starts and ends
        :param seg_lung: segmentation of the lung
        :return: the start of the liver and the end of the liver
        """
        sagittal_vals, coronal_vals, axials_vals = np.where(
            seg_lung != 0)
        axial_up = np.int(np.mean(axials_vals))
        sagittal_vals, coronal_vals, axials_vals = np.where(
            self.aorta_mat != 0)
        axial_down = np.int(np.min(axials_vals))
        seg_body = self.__IsolateBody()
        min_s, min_c, _, max_s, max_c, _ = \
            self.__found_bounding_box( seg_body)
        return min_s, max_s, min_c, max_c, axial_down, axial_up

    def __find_threshold(self, roi_seg):
        """
        Find the threshold for the algorithm
        :return: the th value
        """
        val1 = np.mean(self.ct_mat[np.where(self.l1_mat != 0)])
        val2 = np.mean(self.ct_mat[np.where(roi_seg != 0)])
        th = (np.abs(val2 - val1) * TH_PRE) / 100
        return th

    def __homogeneity_functions(self, shape_mat, curr_new, roi_seg, ct_mat, discovered):
        """
        find new voxels of the liver
        :param shape_mat: the size of the crop
        :param curr_new: the optional candidates
        :param roi_seg: the segmentation
        :param ct_mat: the crop ct scan
        :param discovered: the voxels that part of the liver
        :return: the new voxels that belong the liver
        """
        th = self.__find_threshold(roi_seg)
        candidates = np.zeros((shape_mat))
        candidates[np.where(curr_new != 0)] = np.copy(
            ct_mat[np.where(curr_new != 0)])
        mean_val = np.mean(ct_mat[np.where(discovered != 0)])
        trash = np.abs(candidates - mean_val)
        candidates[trash >= th] = 0
        candidates[trash < th] = 1
        candidates[np.where(curr_new != 1)] = 0

        return candidates

    def __msgr_loop(self, curr, shape_mat, visited, roi_seg, ct_mat, discovered):

        # find optional candidates
        curr_new = binary_dilation(curr, np.ones((5, 5, 5)))
        curr_new = binary_closing(curr_new, np.ones((5, 5, 5)))
        curr_new = np.array(curr_new, dtype=np.int)
        curr = np.zeros((shape_mat))
        curr_new[np.where(visited != 0)] = 0
        visited[np.where(curr_new != 0)] = 1

        # find voxels that belong the liver
        candidates = \
            self.__homogeneity_functions(shape_mat, curr_new, roi_seg, ct_mat,
                                         discovered)
        s_vals, c_vals, a_vals = np.where(candidates != 0)
        curr[s_vals, c_vals, a_vals] = 1
        discovered[s_vals, c_vals, a_vals] = 2

        return curr, visited, discovered

    def __msgr_operations(self, seg):
        """
        apply operations on seg for final results
        :param seg: the segmentation
        :return:segmentation after operations
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

        if '.' + split_nifty[-2] + '.' + split_nifty[-1] != NIFTY_END:
            return FAIL

        file = nib.load(self.dir_input + nifty_file)
        mat = file.get_data()
        return (mat, file)


    def find_ROI_segmentation(self):
        """
        create roi of the liver
        :return: roi of the liver
        """

        seg = self.__IsolateBody()
        seg_lung = self.__isolate_lung(seg)
        box = self.__built_box_roi(seg_lung)
        return box, seg_lung




    def find_origin_segmentation(self, roi_seg, parameters):
        """
        preform segmentation for the origin
        :param roi_seg: segmentation for the roi
        :param parameters: segmentation for the lung
        :return: segmentation for the lung
        """
        seg_lung = parameters[0]
        # find borders to scale down the calculations
        min_s, max_s, min_c, max_c, axial_min, axial_up = self.__find_borders(seg_lung)
        s_vals, c_vals, a_vals = self.__find_seeds(roi_seg[min_s: max_s, min_c: max_c, axial_min:axial_up])

        shape_mat = self.ct_mat[min_s: max_s, min_c: max_c, axial_min:axial_up].shape
        seg_final = np.zeros((self.ct_mat.shape))
        ct_mat = np.copy(self.ct_mat[min_s: max_s, min_c: max_c, axial_min:axial_up])
        min_val = np.min(ct_mat)

        # remove the aorta voxels
        ct_mat[np.where(self.aorta_mat[min_s: max_s, min_c: max_c, axial_min:axial_up] != 0)] = min_val

        # update parameters for
        th = self.__find_threshold(roi_seg)
        curr = np.zeros((shape_mat))
        visited = np.zeros((shape_mat))
        discovered = np.zeros((shape_mat))
        visited[s_vals, c_vals, a_vals] = 1
        curr[s_vals, c_vals, a_vals] = 1
        discovered[s_vals, c_vals, a_vals] = 1
        count = 0
        # TODO recursive iter
        # TODO compare times
        while (np.any(curr)):
            count = count + 1
            if count > NUMBER_OF_IT:
                break

            curr, visited, discovered = self.__msgr_loop(curr, shape_mat, visited, roi_seg, ct_mat,
                        discovered)

        # built final segmentation
        seg_final[min_s: max_s, min_c: max_c, axial_min:axial_up][np.where(discovered != 0)] = \
        discovered[np.where(discovered != 0)]
        discovered[np.where(discovered != 0)] = 5
        seg_final[min_s: max_s, min_c: max_c, axial_min:axial_up][np.where(discovered != 0)] = \
        discovered[np.where(discovered != 0)]
        # apply operations
        seg_final = self.__msgr_operations(seg_final)
        # save final segmentation
        seg_file = nib.Nifti1Image(seg_final, self.ct_file.affine)
        nib.save(seg_file,
                 self.dir_results + self.ct_scan + self.name__results + NIFTY_END)

        return seg_final


    def find_origin_all(self):
        """
        Find roi region for the origin and afterwords finds segmentation
        for the origin and save it in the results path
        """
        seg_roi, seg_lung = self.find_ROI_segmentation()
        self.find_origin_segmentation(seg_roi, [seg_lung])







# -------------------------------- End class ----------------------------------

def check(num_case):
    liver_obj = Liver_Segmentation('Case' + str(num_case) + '_CT.nii.gz',
                                   'Case' + str(num_case) + '_Aorta.nii.gz',
                                   DIR_INPUT,
                                   DIR_RESULTS + '\\' + 'Case' + str(
                                       num_case) + '\\',"resualts_final",
                                   'Case' + str(num_case) + '_L1.nii.gz')

    liver_obj.find_origin_all()




#
if __name__ == '__main__':
    for i in range(1, 2):
        # print(i)
        check(i)

#
