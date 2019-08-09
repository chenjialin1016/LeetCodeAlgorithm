//
//  Array_Algorithm.swift
//  LeetCodeAlgorithm
//
//  Created by Jialin Chen on 2019/8/2.
//  Copyright © 2019年 CJL. All rights reserved.
//

import Cocoa

//与数组相关的算法

class Array_Algorithm {
    
    //MARK:--------两数之和
    /*
     给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
     
     你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
     */
    class func twoSum(_ nums: [Int], _ target: Int)->[Int]{
        /*
         暴力法：遍历数组，并查找是否存在一个值与target-x相等的元素
    
         时间复杂度：O(n^2)
         */
        for i in 0..<nums.count {
            for j in i+1..<nums.count{
                if nums[j] == target-nums[i] {
                    return [i, j]
                }
            }
        }
        return []
    }
    class func twoSum2(_ nums: [Int], _ target: Int)->[Int]{
        /*
         两遍哈希表:
         一个简单的实现使用了两次迭代。在第一次迭代中，我们将每个元素的值和它的索引添加到表中。然后，在第二次迭代中，我们将检查每个元素所对应的目标元素（target−nums[i]）是否存在于表中。注意，该目标元素不能是nums[i] 本身！
         
         时间复杂度：O(n)
         */
        var dic: [Int:Int] = [:]
        for i in 0..<nums.count {
            dic[nums[i]] = i
        }
        
        for i in 0..<nums.count {
            let complement = target - nums[i]
            if dic[complement] != nil && dic[complement] != nums[i]{
                return [i, dic[complement]!]
            }
        }
        return []
    }
    class func twoSum3(_ nums: [Int], _ target: Int)->[Int]{
        /*
         一遍哈希表：
         在进行迭代并将元素插入到表中的同时，我们还会回过头来检查表中是否已经存在当前元素所对应的目标元素。如果它存在，那我们已经找到了对应解，并立即将其返回。
         
         时间复杂度：O(n)
         */
        var dic: [Int:Int] = [:]
        
        for i in 0..<nums.count {
            let complement = target - nums[i]
            if dic[complement] != nil{
                return [i, dic[complement]!]
            }
            dic[nums[i]] = i
        }
        return []
    }

    //MARK:--------寻找两个有序数组的中位数
    /*
     给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。
     请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
     你可以假设 nums1 和 nums2 不会同时为空。
     */
    class func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
        //思路，将两个数组拷贝至同一个数组中，然后排序，找出中位数
        
        var allArr : NSMutableArray = NSMutableArray.init(array: nums1)
        allArr.addObjects(from: nums2)
        //排序
        allArr.sort { (obj1, obj2) -> ComparisonResult in
            let num1 : Int = obj1 as! Int
            let num2 : Int = obj2 as! Int
            if num1 > num2{
                return ComparisonResult.orderedDescending
            }else{
                return ComparisonResult.orderedAscending
            }
        }
        //根据下标找出中位数
        if ((allArr.count-1)%2) == 0{
            return allArr[(allArr.count-1)/2] as! Double
        }else{
            let flag = (allArr.count-1)/2
            return Double((allArr[flag] as! Int)+(allArr[flag+1] as! Int))/2
        }
    }
    class func findMedianSortedArrays2(_ nums1: [Int], _ nums2: [Int]) -> Double {
        //思路--递归法
        
        //         var m = nums1.count
        //         var n = nums2.count
        
        //         var nums1 = nums1
        //         var nums2 = nums2
        
        
        //         if m>n {//如果第一个数组的个数大于第二个，则交换两个数组的顺序,目的是确保第一数组的个数m <= n
        //             let temp : [Int] = nums1
        //             nums1 = nums2
        //             nums2 = temp
        
        //             let tmp = m
        //             m = n
        //             n = tmp
        //         }
        
        //         var iMin = 0, iMax = m, halfLen = (m+n+1)/2
        //         while(iMin <= iMax){
        //             let i = (iMin+iMax)/2
        //             let j = halfLen-i
        //             if (i<iMax)&&(nums2[j-1]>nums1[i]){
        //                 iMin = i+1 //i太小，需要增大
        //             }
        //             else if (i>iMin)&&(nums1[i-1]>nums2[j]){
        //                 iMax = i-1 //i太大，需要减小
        //             }
        //             else{ // i 是完美的
        //                 var maxLeft = 0
        //                 if (i==0){
        //                     maxLeft = nums2[i-1]
        //                 }
        //                 else if (j==0){
        //                     maxLeft = nums1[i-1]
        //                 }
        //                 else{
        //                     maxLeft = max(nums1[i-1], nums2[j-1])
        //                 }
        
        //                 if ((m+n)%2 == 1){
        //                     return Double(maxLeft)
        //                 }
        
        //                 var minRight = 0
        //                 if (i==m) {
        //                     minRight = nums2[j]
        //                 }
        //                 else if (j==n){
        //                     minRight = nums1[i]
        //                 }
        //                 else{
        //                     minRight = min(nums2[j], nums1[i])
        //                 }
        
        //                 return Double(maxLeft+minRight)/2.0
        //             }
        //         }
        //         return 0
        
        let m : Int = nums1.count
        let n : Int = nums2.count
        var left : Int = (m+n+1)/2
        var right : Int = (m+n+2)/2
        return Double(findKth(nums1, 0, nums2, 0, left)+findKth(nums1, 0, nums2, 0, right))/2.0
    }
    //寻找第k个元素
    class func findKth(_ nums1: [Int], _ i : Int, _ nums2: [Int], _ j : Int, _ k : Int)->Int{
        //递归终止条件
        if (i>=nums1.count) {
            return nums2[j+k-1]
        }
        if (j>=nums2.count){
            return nums1[i+k-1]
        }
        if k==1 {
            return min(nums1[i],nums2[j])
        }
        
        
        //判断以i为起始位置，第k/2个元素，是否在数组内
        let n = i+k/2-1
        var midVal1 : Int!
        if n<nums1.count {
            midVal1 = nums1[n]
        }
        else{
            midVal1 = nums1[nums1.count-1]
        }
        
        let m = j+k/2-1
        var midVal2 : Int!
        if m<nums2.count {
            midVal2 = nums2[m]
        }
        else{
            midVal2 = nums2[nums2.count-1]
        }
        
        if midVal1<midVal2{
            if n<nums1.count {
                //移除了最小的一半元素后，k减少k/2
                return findKth(nums1, i+k/2, nums2, j, k-k/2)
            }
            else{
                return nums2[k-nums1.count-i-1]
            }
        }else{
            if m<nums2.count {
                //移除了最小的一半元素后，k减少k/2
                return findKth(nums1, i, nums2, j+k/2, k-k/2)
            }
            else{
                return nums1[k-nums2.count-j-1]
            }
        }
    }
    //MARK:--------盛最多水的容器
    /*
     给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
     说明：你不能倾斜容器，且 n 的值至少为 2。
     */
    class func maxArea(_ height: [Int]) -> Int {
        
        //判断整数个数是否小于2 ==》个数至少为2
        if height.count < 2 {
            return 0
        }
        
        var maxArea = 0, i=0, j = height.count-1
        while i < j {
            maxArea = max(maxArea, min(height[i], height[j])*(j-i))
            if height[i] <= height [j] {
                i += 1
            }else{
                j -= 1
            }
            
        }
        
        return maxArea
    }
    class func maxArea2(_ height: [Int]) -> Int {
        
        //判断整数个数是否小于2 ==》个数至少为2
        if height.count < 2 {
            return 0
        }
        
        var maxArea = 0, i=0, j = height.count-1
        var area = 0
        while i < j {
            area = min(height[i], height[j])*(j-i)
            maxArea = max(maxArea, area)
            if height[i] <= height [j] {
                i += 1
            }else{
                j -= 1
            }
            
        }
        return maxArea
    }
    //MARK:--------三数之和
    /*
     给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
     注意：答案中不可以包含重复的三元组。
     */
    class func threeSum(_ nums: [Int]) -> [[Int]] {
        if nums.count == 0 {
            return []
        }
        
        let nums = nums.sorted { (a, b) -> Bool in
            return a < b
        }
        let len = nums.count
        var array : [[Int]] = [[Int]]()
        for i in 0..<len {
            if i>0 && nums[i] == nums[i-1] {
                continue
            }
            var left = i+1
            var right = len-1
            
            while left < right {
                let tmp = nums[i]+nums[left]+nums[right]
                if tmp==0 {
                    array.append([nums[i], nums[left], nums[right]])
                    while left<right && nums[left]==nums[left+1]{
                        left = left+1
                    }
                    while left<right && nums[right]==nums[right-1]{
                        right = right-1
                    }
                    
                    left = left+1
                    right = right-1
                    
                }else if tmp>0 {
                    //三数之和>0时，将右指针往左移一位
                    right = right - 1
                }else{
                    //三数之和<0时，将左指针往右移一位
                    left = left+1
                }
            }
        }
        
        return array
    }
   
    //MARK:--------最接近的三数之和
    /*
     给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。
     
     例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.
     与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
     
     */
    class func threeSumClosest(_ nums: [Int], _ target: Int) -> Int {
        if nums.count == 0 {
            return 0
        }
        
        //将数组排序
        let nums = nums.sorted { (a, b) -> Bool in
            return a<b
        }
        let len = nums.count
        var res = nums[0]+nums[1]+nums[len-1]
        for i in 0..<len-2 {
            
            if i>0 && nums[i]==nums[i-1]{
                continue
            }
            
            //双指针遍历剩下的两位数
            var left = i+1
            var right = len-1
            
            while left < right{
                let tmp = nums[i]+nums[left]+nums[right]
                if tmp == target {
                    return target
                }
                if abs(res-target) > abs(tmp-target){
                    res = tmp
                }
                if tmp>target {
                    right -= 1
                }
                if tmp<target{
                    left += 1
                }
            }
            
        }
        return res
    }
    class func threeSumClosest2(_ nums: [Int], _ target: Int) -> Int {
        if nums.count == 0 {
            return 0
        }
        
        //将数组排序
        let nums = nums.sorted()
        let len = nums.count
        var res = nums[0]+nums[1]+nums[len-1]
        for i in 0..<len-2 {
            
            if i>0 && nums[i]==nums[i-1]{
                continue
            }
            
            //双指针遍历剩下的两位数
            var left = i+1
            var right = len-1
            
            while left < right{
                let tmp = nums[i]+nums[left]+nums[right]
                if tmp == target {
                    return target
                }
                if abs(res-target) > abs(tmp-target){
                    res = tmp
                }
                if tmp>target {
                    right -= 1
                }
                if tmp<target{
                    left += 1
                }
            }
            
        }
        return res
    }
    //MARK:--------四数之和
    /*
     给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
     注意：答案中不可以包含重复的四元组。
     
     示例： 给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。
     满足要求的四元组集合为：
     [
     [-1,  0, 0, 1],
     [-2, -1, 1, 2],
     [-2,  0, 0, 2]
     ]
     */
    class func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
        /*
         采用4个指针的形式，前两个遍历，后两个用来检验数据是否正确
         1、i指针从0开始到倒数第三个；
         2、j指针从i后面开始到倒数第二个；
         3、left从前往后，right从后往前，活动范围是从j开>始到最后，left和right类似两数之和的问题。
         */
        if nums.count < 4{
            return []
        }
        var nums = nums.sorted()
        var result : [[Int]] = [[Int]]()
        for i in 0..<nums.count-2 {
            //取出指针i可能重复的情况
            if i != 0 && nums[i] == nums[i-1]{
                continue
            }
            for j in i+1..<nums.count{
                // 去除j可能重复的情况
                if j != i+1 && nums[j] == nums[j-1]{
                    continue
                }
                
                var left = j+1
                var right = nums.count-1
                
                while left<right{
                    //不满足条件或者重复的，继续遍历
                    if (left != j+1 && nums[left]==nums[left-1]) || (nums[i]+nums[j]+nums[left]+nums[right]<target){
                        left += 1
                    }else if (right != nums.count-1 && nums[right]==nums[right+1]) || (nums[i]+nums[j]+nums[left]+nums[right]>target){
                        right -= 1
                    }else{
                        var list : [Int] = []
                        list.append(nums[i])
                        list.append(nums[j])
                        list.append(nums[left])
                        list.append(nums[right])
                        
                        result.append(list)
                        
                        //满足条件，进入下一次遍历
                        left += 1
                        right -= 1
                    }
                }
            }
        }
        return result
    }
    //MARK:--------K数之和
    /*
     一个是最基础的 二数之和 问题，这里用的是双指针法（K_Sum_Recursive_Template 里的 if 部分）
     然后就是递归部分，递归直到变成 二数之和 问题（K_Sum_Recursive_Template 里的 else 部分）
     这里提供了一个贼简单的接口：kSum(int[] nums, int target, int k)，复制上去就可以用。
     */
    /**
     * 我是一个接口，在系统提供的他们的方法里面调用我即可
     *
     * 相当加了一层包装，对外提供了一个系统可以使用的接口
     * @param nums 系统给定的数组
     * @param target 系统要求的目标值
     * @return 系统要求的返回值
     */
    class func KSum(_ nums: [Int], _ target: Int, _ k: Int)->[[Int]]{
        //先排序
        var nums = nums.sorted()
        
        //根据模版的要求，将该方法的输入都准备好
        var stack: [Int] = [Int].init(repeating: 0x3f3f3f3f, count: k)
        var stack_index : Int = -1
        var begin = 0
        
        //递归开始
        var ans : [[Int]] = K_Sum_Recursive(nums, &stack, &stack_index, k, begin, target)
        return ans
    }
    /**
     * K数之和问题的模板方法，所有K数问题都可以调用这个方法
     * @param nums 输入的数组
     * @param stack 定义的一个长度为 k_sum 问题中的 k 的数组，初始化为0x3f3f3f3f
     * @param stack_index 栈指针，初始化值为-1
     * @param k 表明当前问题被 分解/递归 成了 k数之和 的问题
     * @param begin 当前问题要固定的值的起点
     * @param target 当前 k数之和 的目标和
     * @return 当前 k数之和 的解集，要在上一层合并到最终解集里去
     */
    private class func K_Sum_Recursive(_ nums: [Int], _ stack: inout [Int], _ stack_index: inout Int, _ k: Int, _ begin: Int, _ target: Int)->[[Int]]{
        
        var ans : [[Int]] = [[Int]]()
        //当递归到两数之和时，不再进行递归，直接使用左右指针法解决
        if k==2 {
            var temp_ans : [Int]
            var left = begin
            var right = nums.count-1
            while left<right{
                if nums[left]+nums[right]>target{
                    //过大，右指针左移
                    right -= 1
                }else if nums[left]+nums[right]<target{
                    //过小，左指针右移
                    left += 1
                }else{
                    //相等，加入序列，左右指针同时向内移动一次
                    temp_ans = [Int]()
                    stack_index += 1
                    stack[stack_index] = nums[left]
                     stack_index += 1
                    stack[stack_index] = nums[right]
                    
                    //当前栈中的元素符合要求，将其加入list中，并将list加入当前问题的解集
                    for i in 0...stack_index{
                        temp_ans.append(stack[i])
                    }
                    ans.append(temp_ans)
                    
                    //栈的清理工作
                    stack[stack_index] = 0x3f3f3f3f
                     stack_index -= 1
                    stack[stack_index] = 0x3f3f3f3f
                    stack_index -= 1
                    
                    left += 1
                    right -= 1
                    while left<right && nums[left]==nums[left-1]{
                        left += 1
                    }
                    while left<right && right<nums.count-1 && nums[right]==nums[right+1]{
                        right -= 1
                    }
                }
            }
        }else{
            var target_temp : Int
            for i in begin..<nums.count-k+1{
                if i > begin && nums[i]==nums[i-1]{
                    continue
                }
                
                //在固定一个数后，问题被降级为一个 k-1数之和 的问题
                //确定 k-1数之和 的目标和
                target_temp = target - nums[i]
                //将当前选定的数字压入栈中，便于最后加入解集中
                stack_index += 1
                stack[stack_index] = nums[i]
                //递归调用 k-1数之和 的问题求解
                var ans_temp = K_Sum_Recursive(nums, &stack, &stack_index, k-1, i+1, target_temp)
                //选定当前数字的情况下，k-1数之和 问题求解完毕
                //将该数字出栈，为选择下一个备选值做准备
                stack[stack_index] = 0x3f3f3f3f
                stack_index -= 1
                //将当前解集加入当前 k数之和 的解集中
                ans.append(contentsOf: ans_temp)
            }
        }
        return ans
    }
    //MARK:--------删除排序数组中的重复项
    /*
     给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
     不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
     
     示例 1:
     给定数组 nums = [1,1,2],
     函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。
     你不需要考虑数组中超出新长度后面的元素。
     */
    class func removeDuplicates(_ nums: inout [Int]) -> Int {
        if nums.count == 0 {
            return 0
        }
        var current : Int = nums[0]
        //需要删除的下标数组
        var idx : [Int] = [current]
        //  遍历
        for i in 1..<nums.count {
            if current == nums[i] {
                continue
            }else{
                idx.append(nums[i])
                current = nums[i]
            }
        }
        nums = idx
        // print(nums)
        return nums.count
    }
    class func removeDuplicates2(_ nums: inout [Int]) -> Int {
        //双指针
        if nums.count == 0 {
            return 0
        }
        //慢指针
        var i = 0
        //j为快指针
        for j in 1..<nums.count {
            if nums[j] != nums[i] {
                i += 1
                nums[i] = nums[j]
            }
        }
        print(nums[0...i])
        return i+1
    }
    //MARK:--------移除元素
    /*
     给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度。
     不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
     元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
     
     示例 1:
     给定 nums = [3,2,2,3], val = 3,
     函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
     你不需要考虑数组中超出新长度后面的元素。
     */
    class func removeElement(_ nums: inout [Int], _ val: Int) -> Int {
        if nums.count == 0 {
            return 0
        }
        var temp : [Int] = [Int]()
        for num in nums {
            if num != val{
                temp.append(num)
            }else{
                continue
            }
        }
        nums = temp
        return nums.count
    }
    class func removeElement2(_ nums: inout [Int], _ val: Int) -> Int {
        /*
         双指针 O(n)
         当nums[j] 与给定的值相等时，递增j 以跳过该元素。只要nums[j]!=val，我们就复制nums[j] 到nums[i] 并同时递增两个索引。重复这一过程，直到j 到达数组的末尾，该数组的新长度为i。
         */
        if nums.count == 0 {
            return 0
        }
       var i = 0
        for j in 0..<nums.count {
            if nums[j] != val{
                nums[i] = nums[j]
                i += 1
            }
        }
        return i
    }
    class func removeElement3(_ nums: inout [Int], _ val: Int) -> Int {
        /*
         双指针 O(n)---当要删除的元素很少时
         当我们遇到nums[i]=val 时，我们可以将当前元素与最后一个元素进行交换，并释放最后一个元素。这实际上使数组的大小减少了 1。
         */
        if nums.count == 0 {
            return 0
        }
        var i = 0
        var n = nums.count
        while i<n {
            if nums[i] == val{
                nums[i] = nums[n-1]
                n -= 1
            }else{
                i += 1
            }
        }
        return n
    }
    //MARK:--------下一次排列
    /*
     实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
     
     如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
     
     必须原地修改，只允许使用额外常数空间。
     
     以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
     1,2,3 → 1,3,2
     3,2,1 → 1,2,3
     1,1,5 → 1,5,1
     */
    class func nextPermutation(_ nums: inout [Int]) {
        /*
         一遍扫描 O(n)
         从右开始遍历数组，找到第一个不按升序排列的数字，记为a[i-1]
         从a[i-1]右边开始遍历数组，找到第一个比a[i-1]大的数字，记为a[j]
         交换a[i-1]~a[j],然后将a[i-1]后面的数字全部翻转
         */
        var i = nums.count-2
        while i>=0 && nums[i+1]<=nums[i] {
            i -= 1
        }
        if i>=0 {
            var j = nums.count-1
            while j>=0 && nums[j]<=nums[i]{
                j -= 1
            }
            nums.swapAt(i, j)
        }
        reverse(&nums, i+1)
        print(nums)
    }
    private class func reverse(_ nums: inout [Int], _ start: Int){
        var i = start
        var j = nums.count-1
        while i<j {
            nums.swapAt(i, j)
            i += 1
            j -= 1
        }
    }
    //MARK:--------搜索旋转排序数组
    /*
     假设按照升序排序的数组在预先未知的某个点上进行了旋转。
     ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
     搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
     你可以假设数组中不存在重复的元素。
     你的算法时间复杂度必须是 O(log n) 级别。
     
     示例 1:
     输入: nums = [4,5,6,7,0,1,2], target = 0
     输出: 4
     */
    class func search(_ nums: [Int], _ target: Int) -> Int {
        //二分查找法：先找到数组中最小的数 即发生旋转的下标
        let n = nums.count
        if n==0 {
            return -1
        }
        if n==1 {
            return nums[0]==target ? 0 : -1
        }
        
        let rotate_index = find_rotate_index(nums, 0, n-1)
        
        //如果target是最小的值
        if nums[rotate_index] == target {
            return rotate_index
        }
        //如果数组并没有旋转，需要查找整个数组
        if rotate_index == 0 {
            return search(nums, target, 0, n-1)
        }
        if target < nums[0] {
            //如果目标值小于第一个数，在旋转下标的右边查找
            return search(nums, target, rotate_index, n-1)
        }else{
            return search(nums, target, 0, rotate_index)
        }
    }
    //找到旋转的下标
    private class func find_rotate_index(_ nums : [Int], _ left: Int, _ right : Int)->Int{
        var left = left
        var right = right
        if nums[left] < nums[right] {
            //数组未发生旋转
            return 0
        }
        
        //二分查找
        while left <= right {
            var pivot = (left+right)/2
            if nums[pivot] > nums[pivot+1] {
                //基准数>基准后一位，往右边查找
                return pivot+1
            }else{
                //如果基准数小于左边左边的数，则右下标左移一位，反之，左下标右移一位
                if nums[pivot]<nums[left]{
                    right = pivot-1
                }else{
                    left = pivot+1
                }
            }
        }
        return 0
    }
    //根据下标二分查找最小的数
    private class func search(_ nums : [Int], _ target : Int, _ left: Int, _ right : Int)->Int{
        var left = left
        var right = right
        
        while left <= right {
            var pivot = (left+right)/2
            if nums[pivot]==target {
                return pivot
            }else{
                if target < nums[pivot] {
                    right = pivot-1
                }else{
                    left = pivot+1
                }
            }
        }
        return -1
    }
    class func search2(_ nums: [Int], _ target: Int) -> Int {
        let n = nums.count
        if n==0 {
            return -1
        }
        var left = 0
        var right = n-1
        while left < right {
            var mid = left + (right-left)/2
            if nums[mid] > nums[right] {
                //如果中间的数大于右边的数，从mid的右边查找
                left = mid+1
            }else{
                right = mid
            }
        }
        //分割点下标
        let split_t = left
        left = 0
        right = n-1
        
        //判断分割点在target的左边还是右边
        if nums[split_t] <= target && target <= nums[right] {
            //分割点在target的右边
            left = split_t
        }else{
            //分割点在target的左边
            right = split_t
        }
        while left <= right {
            let mid = left + (right-left)/2
            if nums[mid] == target {
                //中间值等于目标值
                return mid
            }else if nums[mid] > target {
                //中间值大于目标值，右边下标左移一位
                right = mid - 1
            }else{
                //反之，左边下标右移一位
                left = mid+1
            }
        }
        return -1
    }
    class func search3(_ nums: [Int], _ target: Int) -> Int {
        //直接用二分法，判断二分点
        /*
         1、直接等于target
         2、在左半边递增区域
         1）target 在 left 和 mid 之间
         2）不在之间
         3、在右边的递增区域、
         1）target在mid 和 right 之间
         2）不在之间
         */
        let n = nums.count
        if n==0 {
            return -1
        }
        
        var left = 0
        var right = n-1
        while left < right {
            let mid = left+(right-left)/2
            if nums[mid] == target {
                return mid
            }else if nums[mid] >= nums[left]{//二分点在左半边递增区域
                if nums[left] <= target && target < nums[mid] {
                    //target位于 left 和 mid 之间
                    right = mid-1
                }else{
                    left = mid+1
                }
            }else if nums[mid] < nums[right]{
                //二分点在右边的递增区域
                if nums[mid] < target && target <= nums[right] {
                    left = mid+1
                }else{
                    right = mid-1
                }
            }
        }
        //        print(left, right)
        return nums[left] == target ? left : -1
    }
    //MARK:--------在排序数组中查找元素的第一个和最后一个位置
    /*
     给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
     你的算法时间复杂度必须是 O(log n) 级别。
     如果数组中不存在目标值，返回 [-1, -1]。
     
     示例 1:
     输入: nums = [5,7,7,8,8,10], target = 8
     输出: [3,4]
     */
    class func searchRange(_ nums: [Int], _ target: Int) -> [Int] {
        /*
         线性查找 O(n)
         */
        var range: [Int] = [-1, -1]
        var arr: [Int] = []
        for i in 0..<nums.count {
            if nums[i] == target {
                arr.append(i)
            }
        }
        if arr.count != 0 {
            range =  [arr.first!, arr.last!]
        }
        return range
    }
    class func searchRange2(_ nums: [Int], _ target: Int) -> [Int] {
        /*
         二分查找 O(logn)
         
         首先，为了找到最左边（或者最右边）包含 target 的下标（而不是找到的话就返回 true ），所以算法在我们找到一个 target 后不能马上停止。我们需要继续搜索，直到 lo == hi 且它们在某个 target 值处下标相同。
         left 参数的引入，它是一个 boolean 类型的变量，指示我们在遇到 target == nums[mid] 时应该做什么。如果 left 为 true ，那么我们递归查询左区间，否则递归右区间。考虑如果我们在下标为 i 处遇到了 target ，最左边的 target 一定不会出现在下标大于 i 的位置，所以我们永远不需要考虑右子区间。
         */
        var range: [Int] = [-1, -1]
        let leftIdx = extremeInsertionIndex(nums, target, true)
        
        if leftIdx == nums.count  || nums[leftIdx] != target{
            return range
        }
        range = [leftIdx, extremeInsertionIndex(nums, target, false)-1]
        
        return range
    }
    private class func extremeInsertionIndex(_ nums: [Int], _ target: Int, _ left: Bool)->Int{
        var lo = 0
        var hi = nums.count
        
        while lo<hi {
            let mid = (lo+hi)/2
            if nums[mid] > target || (left && target == nums[mid]){
                hi = mid
            }else{
                lo = mid+1
            }
        }
        return lo
    }
    //MARK:--------搜索插入位置
    /*
     给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
     你可以假设数组中无重复元素。
     示例 1:
     输入: [1,3,5,6], 5
     输出: 2
     
     示例 2:
     输入: [1,3,5,6], 2
     输出: 1
     */
    class func searchInsert(_ nums: [Int], _ target: Int) -> Int {
        //遍历
        var nums = nums
        for i in 0..<nums.count {
            if nums[i] >= target {
                return i
            }
        }
        return nums.count
    }
    class func searchInsert2(_ nums: [Int], _ target: Int) -> Int {
        //二分查找
        var left = 0
        var right = nums.count-1
        while left<=right {
            let mid = (left+right)/2
            if nums[mid] > target{
                right = mid-1
            }else if nums[mid]<target {
                left = mid+1
            }else{
                return mid
            }
        }
        return left
    }
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    
}

//MARK:--------数组算法测试
func array_AlgorithmTest(){
    print("==========数组相关算法==========")
    
    print("两数之和")
    print(Array_Algorithm.twoSum([2, 7, 11, 15], 9))
    print(Array_Algorithm.twoSum2([2, 7, 11, 15], 9))
    print(Array_Algorithm.twoSum3([2, 7, 11, 15], 9))
    print("\n")
    
    print("寻找两个有序数组的中位数")
    print(Array_Algorithm.findMedianSortedArrays([1, 3], [2]))
    print(Array_Algorithm.findMedianSortedArrays2([1, 3], [2]))
    print(Array_Algorithm.findMedianSortedArrays([1, 2], [3, 4]))
    print(Array_Algorithm.findMedianSortedArrays2([1, 2], [3, 4]))
    print("\n")
    
    print("盛最多水的容器")
    print(Array_Algorithm.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    print(Array_Algorithm.maxArea2([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    print("\n")
    
    print("三数之和")
    print(Array_Algorithm.threeSum([-1, 0, 1, 2, -1, -4]))
    print("\n")
    
    print("最接近的三数之和")
    print(Array_Algorithm.threeSumClosest([-1, 2, 1, -4], 1))
    print(Array_Algorithm.threeSumClosest2([-1, 2, 1, -4], 1))
    print("\n")
    
    print("4数之和")
    print(Array_Algorithm.fourSum([1, 0, -1, 0, -2, 2], 0))
    print("\n")
    
    print("k数之和")
    print(Array_Algorithm.KSum([-1, 0, 1, 2, -1, -4], 0, 3))
    print(Array_Algorithm.KSum([1, 0, -1, 0, -2, 2], 0, 4))
    print("\n")
    
    print("删除排序数组中的重复项")
    var arrayRemoveDuplicates = [1, 1, 2]
    var arrayRemoveDuplicates2 = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    print(Array_Algorithm.removeDuplicates(&arrayRemoveDuplicates))
    print(Array_Algorithm.removeDuplicates2(&arrayRemoveDuplicates))
    print(Array_Algorithm.removeDuplicates(&arrayRemoveDuplicates2))
    print(Array_Algorithm.removeDuplicates2(&arrayRemoveDuplicates2))
    print("\n")
    
    print("移除元素")
    var arrayRemoveElement = [3, 2, 2, 3]
    var arrayRemoveElement2 = [0, 1, 2, 2, 3, 0, 4, 2]
    print(Array_Algorithm.removeElement(&arrayRemoveElement, 3))
    print(Array_Algorithm.removeElement2(&arrayRemoveElement, 3))
    print(Array_Algorithm.removeElement3(&arrayRemoveElement, 3))
    print(Array_Algorithm.removeElement(&arrayRemoveElement2, 2))
    print(Array_Algorithm.removeElement2(&arrayRemoveElement2, 2))
    print(Array_Algorithm.removeElement3(&arrayRemoveElement2, 2))
    print("\n")
    
    print("下一个排列")
    var arrayNextPermutation = [1, 2, 3]
    var arrayNextPermutation2 = [3, 2, 1]
    var arrayNextPermutation3 = [1, 1, 5]
    Array_Algorithm.nextPermutation(&arrayNextPermutation)
    Array_Algorithm.nextPermutation(&arrayNextPermutation2)
    Array_Algorithm.nextPermutation(&arrayNextPermutation3)
    print("\n")
    
    print("搜索旋转排序数组")
    print(Array_Algorithm.search([4, 5, 6, 7, 0, 1, 2], 0))
    print(Array_Algorithm.search2([4, 5, 6, 7, 0, 1, 2], 0))
    print(Array_Algorithm.search3([4, 5, 6, 7, 0, 1, 2], 0))
    print("\n")
    
    print("在排序数组中查找元素的第一个和最后一个位置")
    print(Array_Algorithm.searchRange([5, 7, 7, 8, 8, 10], 8))
    print(Array_Algorithm.searchRange([5, 7, 7, 8, 8, 10], 6))
    print(Array_Algorithm.searchRange2([5, 7, 7, 8, 8, 10], 8))
    print(Array_Algorithm.searchRange2([5, 7, 7, 8, 8, 10], 6))
    print("\n")
    
    print("搜索插入数组")
    print(Array_Algorithm.searchInsert([1, 3, 5, 6], 5))
    print(Array_Algorithm.searchInsert([1, 3, 5, 6], 2))
    print(Array_Algorithm.searchInsert2([1, 3, 5, 6], 5))
    print(Array_Algorithm.searchInsert2([1, 3, 5, 6], 2))
    print("\n")
}
