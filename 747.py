class Solution(object):
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if len(nums) == 1:
            return 0
        m = max(nums)
        index = nums.index(m)
        nums.pop(index)

        n = max(nums)
        # print m, n
        if n * 2 <= m:
            return index
        else:
            return -1

