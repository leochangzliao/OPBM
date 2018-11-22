# import cv2
#
#
# img = cv2.imread('/home/leochang/Downloads/PycharmProjects/py-faster-rcnn/'
#                  'data/VOCdevkit2007/VOC2007/JPEGImages/004875.jpg')
# print img.shape
# cv2.imshow('test',img)
# cv2.waitKey(0)

def takeSecond(elem):
    return elem['img_index']


import numpy as np

test = [{'rpn_simu_gt_boxes': np.array([[149.99998474, 84.37499237, 224.99998474, 187.49998474,
                                         1.],
                                        [84.37499237, 159.37498474, 168.74998474, 253.12498474,
                                         1.],
                                        [131.25, 56.24999619, 215.62498474, 149.99998474,
                                         1.]]), 'img_index': 2403},
        {'rpn_simu_gt_boxes': np.array([[125., 93.75, 387.5, 343.75, 1.]]), 'img_index': 565},
        {'rpn_simu_gt_boxes': np.array([[112.5, 105., 375., 315., 1.],
                                        [187.5, 82.5, 427.5, 285., 1.]]), 'img_index': 7073},
        {'rpn_simu_gt_boxes': np.array([[112.5, 112.5, 400., 325., 1.]]), 'img_index': 8811},
        {'rpn_simu_gt_boxes': np.array([[105.44999695, 61.04999924, 366.29998779, 288.6000061,
                                         0.]]), 'img_index': 934},
        {'rpn_simu_gt_boxes': np.array([[149.5, 74.75, 235.75001526, 155.25,
                                         0.],
                                        [385.25, 92., 465.75003052, 172.5,
                                         0.],
                                        [109.25, 115., 362.25, 310.5,
                                         0.],
                                        [373.75, 161., 448.5, 241.50001526,
                                         0.]]), 'img_index': 3735},
        {'rpn_simu_gt_boxes': np.array([[66.40000153, 66.40000153, 325.35998535, 278.88000488,
                                         1.],
                                        [159.36000061, 79.68000031, 245.67999268, 159.36000061,
                                         1.]]), 'img_index': 7919},
        {'rpn_simu_gt_boxes': np.array([[66.40000153, 66.40000153, 325.35998535, 278.88000488,
                                         1.],
                                        [159.36000061, 79.68000031, 245.67999268, 159.36000061,
                                         1.]]), 'img_index': 7919}
        ]


# print test[0],test[1]
# test.sort(key=takeSecond)
# order = [boxes['img_index'] for boxes in test]
# print order
# test1 = test[order.index(7919)]
# test2 = test[order.index(7919)+1]
# # print test
# l1 = test1['rpn_simu_gt_boxes'][0, 4]
# l2 = test2['rpn_simu_gt_boxes'][:, 4]
# print l1, l2,
#
# print (l1 == l2).all()
# if not (l1 == l2).all():
#     print 'dfjafaf'
# random = [{'img_index': 3, 'img': 2}, {'img_index': 2, 'img': 2},
#           {'img_index': 2, 'img': 2}, {'img_index': 1, 'img': [[3, 2], [3, 5]]}]
# # random.remove({'img_index': 2, 'img': 2})
# print random[:][0]['img_index'], {'img_index': 1, 'img': [[3, 2], [3, 5]]} in random
#
# random.sort(key=takeSecond)
#
# print random
# random[1] = {'img_index': 1, 'img': [[3, 2], [3, 5000]]}
# print random
def get_grids(height, width, startx, starty):
    grid_num = 7  # im.shape (rows,columns,channel) with respect to (height,width)
    grid_pixels_height = int(np.floor(1.0 * height / grid_num))
    grid_pixels_width = int(np.floor(1.0 * width / grid_num))
    img_grids = np.zeros((grid_num * grid_num, 4), dtype=np.uint16)
    print grid_pixels_height, grid_pixels_width, len(img_grids)
    index = 0
    for h in range(0, int(height / grid_num) * grid_num, grid_pixels_height):
        for w in range(0, int(width / grid_num) * grid_num, grid_pixels_width):
            img_grids[index] = [w, h,
                                w + grid_pixels_width if (w <= width and width - w > grid_pixels_width and (
                                        index - 6) % 7 is not 0) else width,
                                h + grid_pixels_height if (
                                        h <= height and height - h > grid_pixels_height and index <= 41)
                                else height]
            index += 1
    return img_grids


# print get_grids(21, 23, 0, 0)
# print [i for i in range(0,15/7*7,2)]
# def softmax(a):
#     max_ = np.max(a)
#     return np.exp(a - max_) / np.sum(np.exp(a - max_))
# print softmax([1, 2, 3])
gt_boxes = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10], [7, 45, 9, 10]])


# hyper_proposals = np.zeros((len(gt_boxes) * 3, 4))
# hyper_labels = np.zeros((len(gt_boxes) * 3,))
# index = 4
# proposals = np.array([[2, 2, 2, 2], [3, 3, 3, 3], [1, 2, 3, 4]])
# hyper_proposals[index * 3:index * 3 + 3, :] = proposals
# print hyper_proposals
# hyper_labels[index * 3:index * 3 + 3] = 5
# print hyper_labels
# a = np.array([0, 3, 4])+1
# b = np.array([1, 33, 44])
# print a
# print np.shape(np.equal(a,b).nonzero())[1]

# a = np.zeros((38,10))
# a[10:20,:] =1
# print a


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        last = []
        cur = []
        for s_ in s:
            if len(cur) == 0:
                cur.append(s_)
            else:
                if s_ not in cur:
                    cur.append(s_)
                else:
                    if len(last) <= len(cur):
                        cur.append(s_)
                        last = cur[:-1]
                        index = last.index(s_)
                        cur = last[index + 1:]
                        cur.append(s_)
                    else:
                        index = cur.index(s_)
                        cur = cur[index+1:]
                        cur.append(s_)
        return max(len(last), len(cur))


s = Solution()
print s.lengthOfLongestSubstring('pkjmrovkjouaqnebjfjaut')
