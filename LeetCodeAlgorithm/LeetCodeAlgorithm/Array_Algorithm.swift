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
    //MARK:--------组合总和
    /*
     给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
     candidates 中的数字可以无限制重复被选取。
     说明：
     所有数字（包括 target）都是正整数。
     解集不能包含重复的组合。
     示例 1:
     输入: candidates = [2,3,6,7], target = 7,
     所求解集为:
     [
     [7],
     [2,2,3]
     ]
     */
    static var result = [[Int]]()
    class func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
        /*
         递归回溯：数组先排序
         */
        var arr = [Int]()
        var candi = candidates.sorted()
        findCombinationSum(candi, 0, target, arr)
        return result
    }
    private class func findCombinationSum(_ candidates: [Int], _ start: Int, _ residue: Int, _ arr: [Int]){
        if residue == 0 {
            result.append(arr)
            return
        }
        //基于原数组是排好序的数组的前提，因为如果计算后面的剩余，只会越来越小
        for i in start..<candidates.count {
            if candidates[i] > residue{
                break
            }
            let array = arr + [candidates[i]]
            //因为元素可以重复使用，这里递归传下去的是i而不是i+1
            //residue-candidates[i]表示下一轮的剩余
            findCombinationSum(candidates, i, residue-candidates[i], array)
        }
    }
    
    //MARK:--------组合总和II
    /*
     给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
     candidates 中的每个数字在每个组合中只能使用一次。
     说明：
     所有数字（包括目标数）都是正整数。
     解集不能包含重复的组合。
     示例 1:
     输入: candidates = [10,1,2,7,6,1,5], target = 8,
     所求解集为:
     [
     [1, 7],
     [1, 2, 5],
     [2, 6],
     [1, 1, 6]
     ]
     */
    static var combinationSum2Result = [[Int]]()
    class func combinationSum2(_ candidates: [Int], _ target: Int) -> [[Int]] {
        /*
         排序 + 回溯 + 剪枝
         
         以 target 为根结点，依次减去数组中的数字，直到小于0 或者等于0，把等于0 的结果记录到结果集中。
         
         这道题与上一问的区别在于：
         第 39 题：candidates 中的数字可以无限制重复被选取。
         第 40 题：candidates 中的每个数字在每个组合中只能使用一次。
         
         编码的不同，就在于，下一层递归的起始索引不一样。
         第 39 题：还从候选数组的当前索引值开始。
         第 40 题：从候选数组的当前索引值的下一位开始。
         */
        /*
         递归回溯：数组先排序
         */
        let arr = [Int]()
        let candi = candidates.sorted()
        findCombinationSum2(candi, 0, target, arr)
        return combinationSum2Result
    }
    private class func findCombinationSum2(_ candidates: [Int], _ start: Int, _ residue: Int, _ arr: [Int]){
        if residue == 0 {
            combinationSum2Result.append(arr)
            return
        }
        //基于原数组是排好序的数组的前提，因为如果计算后面的剩余，只会越来越小
        for i in start..<candidates.count {
            if candidates[i] > residue{
                break
            }
            if i > start && candidates[i] == candidates[i-1]{
                continue
            }
            let array = arr + [candidates[i]]
            //因为元素不可以重复使用，这里递归传下去的是i+1
            //residue-candidates[i]表示下一轮的剩余
            findCombinationSum2(candidates, i+1, residue-candidates[i], array)
        }
    }
    //MARK:--------缺失的第一个正数
    /*
     给定一个未排序的整数数组，找出其中没有出现的最小的正整数。
     示例 1:
     输入: [1,2,0]
     输出: 3
     
     示例 2:
     输入: [3,4,-1,1]
     输出: 2
     */
    class func firstMissingPositive(_ nums: [Int]) -> Int {
        /*
         思想：使用索引作为哈希表 以及 元素的符号作为哈希值来实现是否存在的检测
         算法：
         1）检查 1 是否存在于数组中。如果没有，则已经完成，1 即为答案。
         2）如果 nums = [1]，答案即为 2 。
         3）将负数，零，和大于 n 的数替换为 1 。
         4）遍历数组。当读到数字 a 时，替换第 a 个元素的符号。
         注意重复元素：只能改变一次符号。由于没有下标 n ，使用下标 0 的元素保存是否存在数字 n。
         5）再次遍历数组。返回第一个正数元素的下标。
         6）如果 nums[0] > 0，则返回 n 。
         7）如果之前的步骤中没有发现 nums 中有正数元素，则返回n + 1。
         */
        let nums = nums
        let n = nums.count
        //定义一个新数组，最小的数字肯定就在新数组的下标中
        var arr = [Int].init(repeating: 0, count: n+2)
        for num in nums {
            //排除负数和大于输入数组长度的数，因为缺失的正数肯定小雨数组长素+1
            if num>0 && num<=n{
                //新数组下标为1，则表示有这个数
                arr[num] = 1
            }
        }
        //遍历新数组，看看丢一个没有的正整数是谁就行了
        for i in 1..<n+2 {
            if 0==arr[i]{
                return i
            }
        }
        //应对空数组
        return n+1
    }
    class func firstMissingPositive2(_ nums: [Int]) -> Int {
        /*
         思想：桶排序，什么数字就要放在对应的索引上，其它空着就空着
         算法：我们可以把数组进行一次“排序”，“排序”的规则是：如果这个数字 i 落在“区间范围里”，i 就应该放在索引为 i - 1 的位置上，
         1、数字 i 落在“区间范围里”；
         例如：[3, 4, -1, 1]，一共 4 个数字，那么如果这个数组中出现 “1”、“2”、“3”、“4”，就是我们重点要关注的数字了；
         又例如：[7, 8, 9, 11, 12] 一共 5 个数字，每一个都不是 “1”、“2”、“3”、“4”、“5” 中的一个，因此我们无须关注它们；
         2、i 就应该放在索引为i - 1 的位置上；
         这句话也可以这么说 “索引为 i 的位置上应该存放的数字是 i + 1”。
         就看上面那张图，数字1 应该放在索引为0 的位置上，数字3 应该放在索引为2 的位置上，数字4 应该放在索引为3 的位置上。一个数字放在它应该放的位置上，我们就认为这个位置是“和谐”的，看起来“顺眼”的。
         按照以上规则排好序以后，缺失的第1 个正数一下子就看出来了，那么“最不和谐”的数字的索引+1，就为所求。那如果所有的数字都不“和谐”，数组的长度+1 就为所求。
         
         最好的例子：[3,4,-1,1]
         整理好应该是这样：[1,-1,3,4]，
         这里 1，3，4 都在正确的位置上，
         -1 不在正确的位置上，索引是 1 ，所以返回 2
         
         [4,3,2,1] 要变成 [1,2,3,4]，*** Offer 上有类似的问题。
         
         这里负数和大于数组长度的数都是"捣乱项"。
         */
        var nums = nums
        let n = nums.count
        
        for i in 0..<n{
            //前两个是在判断是否成为索引
            //后一个是在判断，例如3在不在索引2上
            //即nums[i] ?= nums[nums[i]-1]这里要特别小心
            while nums[i]>0 && nums[i]<=n && nums[nums[i]-1] != nums[i]{
                //第3个条件不成立的索引部分是 i 和 nums[i]-1
                nums.swapAt(nums[i]-1, i)
            }
        }
        
        for i in 0..<n {
            // [1,-2,3,4]
            // 除了 -2 其它都满足： i+1 = num[i]
            if nums[i]-1 != i{
                return i+1
            }
        }
        return n+1
    }
    //MARK:--------接雨水
    /*
     给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
     */
    class func trap(_ height: [Int]) -> Int {
        /*
         暴力法：O(n^2)
         对于数组中的每个元素，我们找出下雨后水能到达的最高位置，等于两遍最大高度的较小值减去当前高度的值
         算法：
         1）初始化ans = 0
         2）从左往右扫描数组：
            （1）初始化 max_left = 0 和 max_right= 0
            （2）从当前元素向左扫描并更新 max_left = max(max_left, height[j])
            （3）从当前元素向右扫描并更新 max_right = max(max_right, height[j])
            （4）将 min(max_left, max_right)-height[i]累加到 ans
         */
        if height.count == 0{
            return 0
        }
        var ans = 0
        var size = height.count
        var i = 1
        while i<size-1 {
            var max_left = 0, max_right = 0
            var j = i
            while j>=0 {
                max_left = max(max_left, height[j])
                j -= 1
            }
            for j in i..<size{
                max_right = max(max_right, height[j])
            }
            ans += min(max_left, max_right)-height[i]
            
            i += 1
        }
        return ans
    }
    class func trap2(_ height: [Int]) -> Int {
        /*
         动态规划 O(n)
         利用两个数组分别存储左边第i列最高的墙的位置max_left[i] 和右边第i列 的墙的位置max_right[i]
         */
        if height.count == 0{
            return 0
        }
        var ans = 0
        var max_left = [Int].init(repeating: 0, count: height.count)
        var max_right = [Int].init(repeating: 0, count: height.count)
        var i = 1
        while i<height.count {
            max_left[i] = max(max_left[i-1], height[i-1])
            i += 1
        }
        i = height.count-2
        while i>=0  {
            max_right[i] = max(max_right[i+1], height[i+1])
            i -= 1
        }
        i = 1
        while i<height.count {
            let minvalue = min(max_left[i], max_right[i])
            if minvalue>height[i] {
                ans += (minvalue-height[i])
            }
            i += 1
        }
        return ans
    }
    class func trap3(_ height: [Int]) -> Int {
        /*
         双指针 O(n)
         对动态规划的空间复杂度的优化:不使用数组，只用一个元素
         */
        if height.count == 0{
            return 0
        }
        var ans = 0
        var max_left = 0
        var max_right = 0
        var left = 1
        //加右指针进去
        var right = height.count-2
        var i = 1
        while i<height.count-1 {
            //从左到右更
            if height[left-1] < height[right+1] {
                max_left = max(max_left, height[left-1])
                let minvalue = max_left
                if minvalue>height[left] {
                    ans += minvalue-height[left]
                }
                left += 1
            }else{
                 //从右到左更
                max_right = max(max_right, height[right+1])
                let minvalue = max_right
                if minvalue>height[right] {
                    ans += minvalue-height[right]
                }
                right -= 1
            }
           i += 1
        }
        
        return ans
    }
    class func trap4(_ height: [Int]) -> Int {
        /*
         栈 O(n)
         总体的原则就是:
         1)当前高度小于等于栈顶高度，入栈，指针后移。
         2)当前高度大于栈顶高度，出栈，计算出当前墙和栈顶的墙之间水的多少，然后计算当前的高度和新栈的高度的关系，重复第 2 步。直到当前墙的高度不大于栈顶高度或者栈空，然后把当前墙入栈，指针后移。
         */
        if height.count == 0{
            return 0
        }
        var ans = 0
        var stack = [Int]()
        var current = 0
        while current<height.count {
            //如果栈不空且当前指向的高度大于栈顶高度就一直循环
            while (stack.count != 0) && height[current]>height[stack.last!] {
                //取出要出栈的元素
                let h = height[stack.last!]
                stack.removeLast()
                //栈空就出去
                if stack.count == 0{
                    break
                }
                //两堵墙之前的距离
                let distance = current - stack.last! - 1
                let minvalue = min(height[stack.last!], height[current])
                ans += distance*(minvalue-h)
            }
            //当前指向的墙入栈
            stack.append(current)
            //指针后移
            current += 1
        }
        
        return ans
    }
    //MARK:--------字符串相乘
    /*
     给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
     示例 1:
     输入: num1 = "2", num2 = "3"
     输出: "6"
     
     示例 2:
     输入: num1 = "123", num2 = "456"
     输出: "56088"
     
     说明：
     num1 和 num2 的长度小于110。
     num1 和 num2 只包含数字 0-9。
     num1 和 num2 均不以零开头，除非是数字 0 本身。
     不能使用任何标准库的大数类型（比如 BigInteger）或直接将输入转换为整数来处理。
     */
    //模拟乘法,首先要进行字符串翻转,让低位在前面,方便处理进位
    class func multiply(_ num1: String, _ num2: String) -> String {
        
        if num1=="0" || num2=="0" {
            return "0"
        }
        
        var res : [Int] = [Int].init(repeating: 0, count: num1.count+num2.count)
        var num1Arr : [Character] = []
        var num2Arr : [Character] = []
        
        for i in num1{
            num1Arr.append(i)
        }
        for j in num2 {
            num2Arr.append(j)
        }
        //字符数组倒序存储
        num1Arr = num1Arr.reversed()
        num2Arr = num2Arr.reversed()
        
        print(num1Arr)
        print(num2Arr)
        
        //遍历数组
        for i in 0..<num1Arr.count {
            for j in 0..<num2Arr.count{
                //记录两两相乘的结果
                res[i+j] += stringToInt(String(num1Arr[i]))*stringToInt(String(num2Arr[j]))
            }
        }
        //进位值
        var carrys = 0
        //将两两相乘的结果遍历
        for i in 0..<res.count {
            res[i] += carrys
            carrys = res[i]/10
            res[i] %= 10
        }
        if carrys != 0 {
            res[res.count-1] = carrys
        }
        //结果值正序
        res = res.reversed()
        
        var str : String = ""
        var i = res[0]==0 ? 1 : 0
        for _ in i..<res.count {
            str += String(res[i])
            
            i += 1
        }
        
        return str
        
    }
    
    class func multiply2(_ num1: String, _ num2: String) -> String {
        
        if num1=="0" || num2=="0" {
            return "0"
        }
        
        var res : [Int] = [Int].init(repeating: 0, count: num1.count+num2.count)
        
        var num1Arr : [Int] = stringToArr(num1)
        var num2Arr : [Int] = stringToArr(num2)
        print(num1Arr)
        print(num2Arr)
        
        //遍历数组
        for i in 0..<num1Arr.count {
            for j in 0..<num2Arr.count{
                //记录两两相乘的结果
                res[i+j] += num1Arr[i]*num2Arr[j]
            }
        }
        //进位值
        var carrys = 0
        //将两两相乘的结果遍历
        for i in 0..<res.count {
            res[i] += carrys
            carrys = res[i]/10
            res[i] %= 10
        }
        if carrys != 0 {
            res[res.count-1] = carrys
        }
        //结果值正序
        res = res.reversed()
        
        var str : String = ""
        var i = res[0]==0 ? 1 : 0
        for _ in i..<res.count {
            str += String(res[i])
            
            i += 1
        }
        
        return str
        
    }
    
    //将字符串转换成数组并反转顺序
    private class func stringToArr(_ str : String)->[Int]{
        var result : [Int] = []
        for i in str{
            result.append(stringToInt(String(i)))
        }
        //字符数组倒序存储
        result = result.reversed()
        return result
    }
    
    //字符利用ascii转换为整数值
    private class func stringToInt(_ char : String)->Int{
        var result : Int = 0
        var result1 : Int = 0
        
        for characterInt in char.unicodeScalars {
            result = Int(characterInt.value)
        }
        for characterInt in "0".unicodeScalars {
            result1 = Int(characterInt.value)
        }
        //        print(result-result1)
        return result-result1
    }
    //MARK:--------通配符匹配
    /*
     给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。
     '?' 可以匹配任何单个字符。
     '*' 可以匹配任意字符串（包括空字符串）。
     两个字符串完全匹配才算匹配成功。
     
     说明:
     s 可能为空，且只包含从 a-z 的小写字母。
     p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。
     */
    class func isMatch(_ s: String, _ p: String) -> Bool {
        /*
         直接比较
         */
       //直接比较，相等，返回
        if s == p {
            return true
        }else{
            //不想等，则可能含有其他特殊符号 ？ 、 *等
            if s.count == p.count && !p.contains("*"){
                //长度相等，且不含*，则含有？
                if p.contains("?"){
                    //遍历比较
                    for i in 0..<p.count{
                        let index = s.index(s.startIndex, offsetBy: i)
                        if s[index] != p[index] && p[index] != "?" {
                            ///比较原则，如果每个字符不想等，且不是？，则不匹配
                            return false
                        }
                    }
                    return true
                }else{
                    //不含？，肯定不匹配
                    return false
                }
            }else{
                //长度不相等或者含有*，如果不含*，肯定不匹配
                if p.contains("*"){
                    //含有*
                    let strings = p.components(separatedBy: "*")
                    var tempStrings = strings
                    tempStrings.removeAll(where: {$0 == ""})
                    if tempStrings.count == 0 {
                        //如果含有空字符串，则说明含有*，肯定匹配
                        return true
                    }
                    //开始比较的位置，接着上一次比较
                    var startCompare = 0
                    var k = 0
                    //是否以*开头
                    var isStartFirst = false
                    
                    //遍历子字符串组成的数组比较
                    for compoment in strings {
                        //开始表交的位置+剩余比价长度>s.count，肯定不匹配
                        if startCompare+compoment.count > s.count{
                            return false
                        }
                        //空字符，说明以*开头或者结尾
                        if compoment.count == 0{
                            //以*开头，标记
                            if k == 0{
                                isStartFirst = true
                            }
                            //以*结尾，走到这说明前面的已经匹配，肯定匹配
                            if k == strings.count-1 {
                                return true
                            }
                            k += 1
                            continue
                        }
                        //如果子字符串的疏朗不等于1，则最后一个子字符串需要跟最后对应的子字符串屁诶批
                        if k>0 && k==strings.count-1 {
                            startCompare = s.count-compoment.count
                        }
                        var temps = s
                        //剔除掉已经比较的字符串
                        if startCompare != 0{///剔除掉已经比较了的字符串
                            let startI = s.index(s.startIndex, offsetBy: startCompare - 1)
                            temps.removeSubrange(s.startIndex...startI)
                        }
                        
                        if !(temps.contains(compoment)){///剩余未匹配的字符串，不包含子字符串
                            if compoment.contains("?") {///如果包含？
                                for i in startCompare..<s.count{///遍历比较，比较原则，字符不想等，且不为？，则不匹配，更新开始比较的位置，继续比较
                                    var isMatch = true
                                    for j in 0..<compoment.count{
                                        if i + j > s.count - 1{
                                            return false
                                        }
                                        let sindex = s.index(s.startIndex, offsetBy: i+j)
                                        let cindex = compoment.index(compoment.startIndex, offsetBy: j)
                                        if compoment[cindex] != s[sindex] && compoment[cindex] != "?"{
                                            if i >= s.count - compoment.count || (!isStartFirst && startCompare == 0) {///如果是第一组，且不是以*开头，则肯定不匹配；如果是最后一组，也肯定不匹配
                                                return false
                                            }
                                            
                                            isMatch = false///不匹配，跳出循环
                                            break
                                        }
                                    }
                                    if isMatch {///匹配到，更新开始匹配位置，跳出循环
                                        startCompare = i + compoment.count
                                        break
                                    }
                                }
                                
                            }else{///如果不包含？，肯定不匹配
                                return false
                            }
                        }else{///剩余未匹配的字符串，包含子字符串，查询位置
                            
                            let range = temps.range(of: compoment)
                            if startCompare == 0 && !isStartFirst{///第一组，且不以*开头
                                let lower = temps.distance(from: s.startIndex, to: range!.lowerBound)
                                if lower != 0{///匹配到的位置不是0，则肯定不匹配
                                    return false
                                }
                            }
                            let up = range!.upperBound
                            startCompare += temps.distance(from: s.startIndex, to: up)///更新开始匹配的位置
                            
                        }
                        k += 1
                    }
                    return true
                    
                }else{///不含*，肯定不匹配
                    return false
                }
            }
        }
    }
    class func isMatch2(_ s: String, _ p: String) -> Bool {
        /*
         双指针：利用两个指针进行遍历
         */
        var sn = s.count
        var pn = p.count
        var i = 0, j = 0, start = -1, match = 0
        while i<sn {
            if j<pn && (s[s.index(s.startIndex, offsetBy: i)] == p[p.index(p.startIndex, offsetBy: j)] || p[p.index(p.startIndex, offsetBy: j)]=="?"){
                i += 1
                j += 1
            }else if j<pn && p[p.index(p.startIndex, offsetBy: j)]=="*"{
                start = j
                match = i
                j += 1
            }else if start != -1{
                j = start+1
                match += 1
                i = match
            }else{
                return false
            }
        }
        while j<pn {
            if p[p.index(p.startIndex, offsetBy: j)] != "*"{
                return false
            }
            j += 1
        }
        return true
    }
    class func isMatch3(_ s: String, _ p: String) -> Bool {
        /*
         动态规划
         
         dp[i][j]表示s到i位置,p到j位置是否匹配!
         初始化:
         dp[0][0]:什么都没有,所以为true
         第一行dp[0][j],换句话说,s为空,与p匹配,所以只要p开始为*才为true
         第一列dp[i][0],当然全部为False
         动态方程:
         如果(s[i] == p[j] || p[j] == "?") && dp[i-1][j-1] ,有dp[i][j] = true
         如果p[j] == "*" && (dp[i-1][j] = true || dp[i][j-1] = true)有dp[i][j] = true
         note:
         ​    ​    dp[i-1][j],表示*代表是空字符,例如ab,ab*
         ​    ​    dp[i][j-1],表示*代表非空任何字符,例如abcd,ab*
         */
        var dp = [[Bool]].init(repeating: [Bool].init(repeating: false, count: p.count+1), count: s.count+1)
        dp[0][0] = true
        var j = 1
        while j<p.count+1 {
            if p[p.index(p.startIndex, offsetBy: j-1)] == "*"{
                dp[0][j] = dp[0][j-1]
            }
            j += 1
        }
        var i = 1
        while i<s.count+1 {
            j = 1
            while j<p.count+1 {
                if s[s.index(s.startIndex, offsetBy: i-1)]==p[p.index(p.startIndex, offsetBy: j-1)] || p[p.index(p.startIndex, offsetBy: j-1)]=="?"{
                    dp[i][j] = dp[i-1][j-1]
                }else if p[p.index(p.startIndex, offsetBy: j-1)] == "*"{
                    dp[i][j] = dp[i][j-1] || dp[i-1][j]
                }
                j += 1
            }
            i += 1
        }
        return dp[s.count][p.count]
    }
    //MARK:--------跳跃游戏II
    /*
     给定一个非负整数数组，你最初位于数组的第一个位置。
     数组中的每个元素代表你在该位置可以跳跃的最大长度。
     你的目标是使用最少的跳跃次数到达数组的最后一个位置。
     示例:
     输入: [2,3,1,1,4]
     输出: 2
     解释: 跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
     说明:
     假设你总是可以到达数组的最后一个位置。
     */
    class func jump(_ nums: [Int]) -> Int {
        /*
         顺藤摸瓜：在每次可跳范围内选择可以使得跳的更远的位置
         O(n)
         */
        var end = 0, maxPosition = 0, steps = 0
        for i in 0..<nums.count-1 {
            //找能跳的更远的
            maxPosition = max(maxPosition, nums[i]+i)
            //遇到边界就更新边界，并且步数加1
            if i == end{
                end = maxPosition
                steps += 1
            }
        }
        return steps
    }
    class func jump2(_ nums: [Int]) -> Int {
        /*
         顺瓜摸藤：我们知道最终要到达最后一个位置，然后我们找前一个位置，遍历数组，找到能到达它的位置，离她最远的就是要找的位置，然后继续找上上个位置，最后到了第0个位置就结束了
         至于离它最远的位置，其实我们从左到右遍历数组，第一个满足的位置就是我们要找的
         O(n^2)
         */
        //要找的位置
        var position = nums.count-1, steps = 0
        //是否到了第0个位置
        while position != 0 {
            for i in 0..<position{
                if nums[i] >= position-i{
                    //更新要找的位置
                    position = i
                    steps += 1
                    break
                }
            }
        }
        return steps
    }
    //MARK:--------全排列
    /*
     给定一个没有重复数字的序列，返回其所有可能的全排列。
     示例:
     输入: [1,2,3]
     输出:
     [
     [1,2,3],
     [1,3,2],
     [2,1,3],
     [2,3,1],
     [3,1,2],
     [3,2,1]
     ]
     */
    class func permute(_ nums: [Int]) -> [[Int]] {
        var output : [[Int]] = []
        var num_list : [Int] = []
        for num in nums {
            num_list.append(num)
        }
        
        let n = nums.count
        backTrack(n, num_list, &output, 0)
        return output
        
    }
    private class func backTrack(_ n : Int, _ nums : [Int], _ output : inout [[Int]], _ first : Int){
        //如果第一个索引为n 即当前排列已完成
        var nums = nums
        if first==n {
            output.append(nums)
        }
        
        for i in first..<n {
            nums.swapAt(first, i)
            backTrack(n, nums, &output, first+1)
            //回溯
            nums.swapAt(first, i)
        }
    }
    //MARK:--------旋转图像
    /*
     给定一个 n × n 的二维矩阵表示一个图像。
     将图像顺时针旋转 90 度。
     说明：
     你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。
     示例 1:
     给定 matrix =
     [
     [1,2,3],
     [4,5,6],
     [7,8,9]
     ],
     原地旋转输入矩阵，使其变为:
     [
     [7,4,1],
     [8,5,2],
     [9,6,3]
     ]
     */
    class func rotate(_ matrix: inout [[Int]]) {
        /*
         转置加翻转:先转置矩阵，然后翻转每一行
         O(n^2)
         */
        let n = matrix.count
        //转置矩阵
        for i in 0..<n {
            for j in i..<n{
                let temp = matrix[j][i]
                matrix[j][i] = matrix[i][j]
                matrix[i][j] = temp
            }
        }
        //翻转每一行
        for i in 0..<n {
            for j in 0..<n/2{
                let temp = matrix[i][j]
                matrix[i][j] = matrix[i][n-j-1]
                matrix[i][n-j-1] = temp
            }
        }
        print(matrix)
    }
    class func rotate2(_ matrix: inout [[Int]]) {
        /*
         旋转四个矩形：使用两个矩阵操作，但是只使用一次操作的方法完成旋转
         O(n^2)
         */
        let n = matrix.count
        for i in 0..<n/2+n%2 {
            for j in 0..<n/2{
                var temp = [Int].init(repeating: 0, count: 4)
                var row = i, col = j
                for k in 0..<4{
                    temp[k] = matrix[row][col]
                    var x = row
                    row = col
                    col = n-1-x
                }
                for k in 0..<4{
                    matrix[row][col] = temp[(k+3)%4]
                    let x = row
                    row = col
                    col = n-1-x
                }
            }
        }
        print(matrix)
    }
    class func rotate3(_ matrix: inout [[Int]]) {
        /*
         旋转四个矩形：在单次循环中旋转4个矩形
         O(n^2)
         */
        let n = matrix.count
        for i in 0..<(n+1)/2{
            for j in 0..<n/2{
                let temp = matrix[n-j-1][i]
                matrix[n-j-1][i] = matrix[n-i-1][n-j-1]
                matrix[n-i-1][n-j-1] = matrix[j][n-i-1]
                matrix[j][n-i-1] = matrix[i][j]
                matrix[i][j] = temp
            }
        }
        print(matrix)
    }
    //MARK:--------字母异位词分组
    /*
     给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
     示例:
     输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
     输出:
     [
     ["ate","eat","tea"],
     ["nat","tan"],
     ["bat"]
     ]
     说明：
     所有输入均为小写字母。
     不考虑答案输出的顺序。
     */
    class func groupAnagrams(_ strs: [String]) -> [[String]] {
        /*
         排序数组分类：当且仅当他们的排序字符相等时，两个字符串是字母异位词
         
         时间复杂度：O(NKlogK)，其中N 是 strs 的长度，而K 是 strs 中字符串的最大长度。当我们遍历每个字符串时，外部循环具有的复杂度为O(N)。然后，我们在O(KlogK) 的时间内对每个字符串排序。
         */
        var result = [[String]]()
        if strs.count == 0{
            return result
        }
        var dic = [String: [String]]()
        for s in strs {
            //排序
            let ca = String(s.sorted())
            //添加到对应的类中
            if !dic.keys.contains(ca){
                var array = [String]()
                array.append(s)
                dic[ca] = array
            }else{
               dic[ca]?.append(s)
            }
        }
        
        for (_, value) in dic{
            result.append(value)
        }
        return result
    }
    class func groupAnagrams2(_ strs: [String]) -> [[String]] {
        /*
         算术基本定理，又称为正整数的唯一分解定理，即：每个大于1的自然数，要么本身就是质数，要么可以写为2个以上的质数的积，而且这些质因子按大小排列之后，写法仅有一种方式。
         利用这个，我们把每个字符串都映射到一个正数上。
         用一个数组存储质数 prime = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103}。
         然后每个字符串的字符减去 ' a ' ，然后取到 prime 中对应的质数。把它们累乘。
         例如 abc ，就对应 'a' - 'a'， 'b' - 'a'， 'c' - 'a'，即 0, 1, 2，也就是对应素数 2 3 5，然后相乘 2 * 3 * 5 = 30，就把 "abc" 映射到了 30。
         
         时间复杂度：O（n∗K），K 是字符串的最长长度。
         */
        var result = [[String]]()
        if strs.count == 0{
            return result
        }
        var dic = [Int: [String]]()
        //每个字母对应一个质数
        let prime: [Int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
        for i in 0..<strs.count{
            var key = 1
            //累乘得到key
            for j in 0..<strs[i].count {
                key *= prime[characterTwoSubInt(String(strs[i][strs[i].index(strs[i].startIndex, offsetBy: j)]), "a")]
            }
            if dic.keys.contains(key) {
                dic[key]?.append(strs[i])
            }else{
                var temp = [String]()
                temp.append(strs[i])
                dic[key] = temp
            }
        }
        
        for (_, value) in dic{
            result.append(value)
        }
        return result
    }
    class func groupAnagrams3(_ strs: [String]) -> [[String]] {
        /*
         按计数分类：当且仅当它们的字符计数（每个字符的出现次数）相同时，两个字符串是字母异位词。
         首先初始化 key = "0#0#0#0#0#"，数字分别代表 abcde 出现的次数，# 用来分割。
         这样的话，"abb" 就映射到了 "1#2#0#0#0"。
         "cdc" 就映射到了 "0#0#2#1#0"。
         "dcc" 就映射到了 "0#0#2#1#0"。
         时间复杂度： O(NK)，其中N 是 strs 的长度，而K 是 strs 中字符串的最大长度。计算每个字符串的字符串大小是线性的，我们统计每个字符串。
         */
        var result = [[String]]()
        if strs.count == 0{
            return result
        }
        var dic = [String: [String]]()
        for i in 0..<strs.count {
            var num = [Int].init(repeating: 0, count: 26)
            //记录每个字符的次数
            for j in 0..<strs[i].count{
                num[characterTwoSubInt(String(strs[i][strs[i].index(strs[i].startIndex, offsetBy: j)]), "a")] += 1
            }
            //转成0#2#2#类似的形式
            var key = ""
            for j in 0..<num.count{
                key = key + String(num[j]) + "#"
            }
            if dic.keys.contains(key){
                dic[key]?.append(strs[i])
            }else{
                var temp = [String]()
                temp.append(strs[i])
                dic[key] = temp
            }
        }
        for (_, value) in dic{
            result.append(value)
        }
        return result
    }
    //MARK:--------pow(x,n)
    /*
     实现 pow(x, n) ，即计算 x 的 n 次幂函数。
     示例 1:
     输入: 2.00000, 10
     输出: 1024.00000
     示例 2:
     输入: 2.10000, 3
     输出: 9.26100
     示例 3:
     输入: 2.00000, -2
     输出: 0.25000
     解释: 2-2 = 1/22 = 1/4 = 0.25
     说明:
     -100.0 < x < 100.0
     n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。
     */
    class func myPow(_ x: Double, _ n: Int) -> Double {
        /*
         利用swift中自带的pow函数
         */
        return pow(x, Double(n))
    }
    class func myPow2(_ x: Double, _ n: Int) -> Double {
        /*
         暴力法：直接模拟该过程，将x连乘n次
         时间复杂度：O(n). 我们需要将 x 连乘 n 次。
         */
        var x = x, n = n
        if n<0{
            x = 1/x
            n = -n
        }
        var ans : Double = 1
        for _ in 0..<n{
            ans *= x
        }
        return ans
    }
    class func myPow3(_ x: Double, _ n: Int) -> Double {
        /*
         快速幂算法(递归)：不需要将 x 再乘 n 次。使用公式（x^n）^2 = x^(2*n) ，我们可以在一次计算内得到x^(2*n) 的值
         时间复杂度: O(logn)
         */
        var x = x, n = n
        if n<0{
            x = 1/x
            n = -n
        }
        return fastPow(x, n)
    }
    private class func fastPow(_ x: Double, _ n:Int)->Double{
        if n == 0{
            return 1.0
        }
        var half:Double = fastPow(x, n/2)
        if n%2 == 0 {
            return half*half
        }else{
            return half*half*x
        }
    }
    class func myPow4(_ x: Double, _ n: Int) -> Double {
        /*
         快速幂算法(循环)：使用公式x^（a+b） = x^a+x^b ，将n看作一系列正整数之和， n=∑ibi，如果可以快速得到x^bi的结果，计算x^n的总时间将被降低
         时间复杂度: O(logn)
         */
        var x = x, n = n
        if n<0{
            x = 1/x
            n = -n
        }
        var ans: Double = 1
        var current_product = x
        var i = n
        while i>0 {
            if i%2 == 1 {
                ans *= current_product
            }
            current_product *= current_product
            i /= 2
        }
        return ans
    }
    //MARK:--------N皇后
    /*
     n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
     给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。
     每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
     示例:
     输入: 4
     输出: [
     [".Q..",  // 解法 1
     "...Q",
     "Q...",
     "..Q."],
     
     ["..Q.",  // 解法 2
     "Q...",
     "...Q",
     ".Q.."]
     ]
     解释: 4 皇后问题存在两个不同的解法。
     */
    static var rows: [Int] = [Int]()
    //主对角线
    static var hills: [Int]!
    //副对角线
    static var dales: [Int]!
    static var N : Int = 0
    static var output: [[String]]!
    static var queens: [Int]!
    class func solveNQueens(_ n: Int) -> [[String]] {
        /*
         回溯：一行可能只有一个皇后且一列也只可能有一个皇后，所以只需按列循环即可
             对于所有的主对角线有 行号+列号 = 常数，对于所有的次对角线 行号-列号=常数，所以可以标记已经在攻击范围下的对角线并且检查一个方格（行号、列号）是否处于攻击位置
         算法：现在已经可以写回溯函数 backtrack(row = 0).
                1）从第一个 row = 0 开始.
                2）循环列并且试图在每个 column 中放置皇后.
                    （1）如果方格 (row, column) 不在攻击范围内
                            （a）在 (row, column) 方格上放置皇后。
                            （b）排除对应行，列和两个对角线的位置。
                            （c）If 所有的行被考虑过，row == N
                                意味着我们找到了一个解
                            （d）Else
                                继续考虑接下来的皇后放置 backtrack(row + 1).
                            （e）回溯：将在 (row, column) 方格的皇后移除.
         时间复杂度：O(N!). 放置第 1 个皇后有 N 种可能的方法，放置两个皇后的方法不超过 N (N - 2) ，放置 3 个皇后的方法不超过 N(N - 2)(N - 4) ，以此类推。总体上，时间复杂度为O(N!) .
         */
        N = n
        rows = [Int].init(repeating: 0, count: n)
        hills = [Int].init(repeating: 0, count: 4*n-1)
        dales = [Int].init(repeating: 0, count: 2*n-1)
        queens = [Int].init(repeating: 0, count: n)
        output = [[String]]()
        
        backTrace(0)
        return output
    }
    private class func isNotUnderAttack(_ row: Int, _ col: Int)->Bool{
        var res = rows[col]+hills[row-col+2*N]+dales[row+col]
        return ((res==0) ? true : false)
    }
    private class func placeQueen(_ row: Int, _ col: Int){
        queens[row] = col
        rows[col] = 1
        hills[row-col+2*N] = 1
        dales[row+col] = 1
    }
    private class func removeQueen(_ row: Int, _ col: Int){
        queens[row] = 0
        rows[col] = 0
        hills[row-col+2*N] = 0
        dales[row+col] = 0
    }
    private class func addSolution(){
        var solution: [String] = [String]()
        for i in 0..<N{
            let col = queens[i]
            var sb = ""
            for j in 0..<col{
                sb += "."
            }
            sb += "Q"
            for j in 0..<N-col-1{
                sb += "."
            }
            solution.append(sb)
        }
        output.append(solution)
    }
    private class func backTrace(_ row: Int){
        for col in 0..<N{
            if isNotUnderAttack(row, col){
                placeQueen(row, col)
                if row+1 == N {
                    addSolution()
                }else{
                    backTrace(row+1)
                }
                removeQueen(row, col)
            }
        }
    }
    //MARK:--------N皇后II
    /*
     n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
     给定一个整数 n，返回 n 皇后不同的解决方案的数量。
     示例:
     输入: 4
     输出: 2
     解释: 4 皇后问题存在如下两个不同的解法。
     [
       [".Q..",  // 解法 1
         "...Q",
         "Q...",
         "..Q."],
     
       ["..Q.",  // 解法 2
         "Q...",
         "...Q",
         ".Q.."]
     ]
     */
    class func totalNQueens(_ n: Int) -> Int {
        /*
         回溯法
         1）从第一个 row = 0 开始.
         2）循环列并且试图在每个 column 中放置皇后.
            （1）如果方格 (row, column) 不在攻击范围内
                在 (row, column) 方格上放置皇后。
                排除对应行，列和两个对角线的位置。
                If 所有的行被考虑过，row == N
                    意味着我们找到了一个解
                Else
                    继续考虑接下来的皇后放置 backtrack(row + 1).
                回溯：将在 (row, column) 方格的皇后移除.
         时间复杂度 O(N!). 放置第 1 个皇后有 N 种可能的方法，放置两个皇后的方法不超过 N (N - 2) ，放置 3 个皇后的方法不超过 N(N - 2)(N - 4) ，以此类推。总体上，时间复杂度为O(N!) .
         */
        let rows = [Int].init(repeating: 0, count: n)
        let hills = [Int].init(repeating: 0, count: 4*n-1)
        let dales = [Int].init(repeating: 0, count: 2*n-1)
        return backTrace(0, 0, n, rows, hills, dales)
    }
    private class func is_not_under_attack(_ row: Int, _ col: Int, _ n: Int, _ rows: [Int], _ hills: [Int], _ dales: [Int])->Bool{
        let res = rows[col]+hills[row-col+2*n]+dales[row+col]
        return (res==0) ? true : false
    }
    private class func backTrace(_ row: Int, _ count: Int, _ n: Int, _ rows: [Int], _ hills: [Int], _ dales: [Int])->Int{
        var rows = rows, hills = hills, dales = dales, count = count
        for col in 0..<n {
            if is_not_under_attack(row, col, n, rows, hills, dales){
                rows[col] = 1
                hills[row-col+2*n] = 1
                dales[row+col] = 1
                
                if row+1==n {
                    count += 1
                }else{
                    count = backTrace(row+1, count, n, rows, hills, dales)
                }
                
                //移除皇后
                rows[col] = 0
                hills[row-col+2*n] = 0
                dales[row+col] = 0
            }
        }
        return count
    }
    class func totalNQueens2(_ n: Int) -> Int {
        /*
         使用bitmap回溯：使用的是位运算
         */
        return backTrace2(0, 0, 0, 0, 0, n)
    }
    private class func backTrace2(_ row: Int, _ hills: Int, _ next_row: Int, _ dales: Int, _ count: Int, _ n: Int)->Int{
        /**
         row: 当前放置皇后的行号
         hills: 主对角线占据情况 [1 = 被占据，0 = 未被占据]
         next_row: 下一行被占据的情况 [1 = 被占据，0 = 未被占据]
         dales: 次对角线占据情况 [1 = 被占据，0 = 未被占据]
         count: 所有可行解的个数
         */
        
        // 棋盘所有的列都可放置，
        // 即，按位表示为 n 个 '1'
        // bin(cols) = 0b1111 (n = 4), bin(cols) = 0b111 (n = 3)
        // [1 = 可放置]
        let columns = (1<<n)-1
        var count = count
        
        //如果已经放置了n个皇后
        if row == n {
            //累加可行解
            count += 1
        }else{
            // 当前行可用的列
            // ! 表示 0 和 1 的含义对于变量 hills, next_row and dales的含义是相反的
            // [1 = 未被占据，0 = 被占据]
            var free_columns = columns & ~(hills | next_row | dales)
            
            //可以找到放置下一个皇后的列
            while free_columns != 0 {
                // free_columns 的第一个为 '1' 的位
                // 在该列我们放置当前皇后
                let curr_column = -free_columns&free_columns
                
                // 放置皇后
                // 并且排除对应的列
                free_columns ^= curr_column
                
                count = backTrace2(row+1, (hills | curr_column)<<1, next_row|curr_column, (dales | curr_column)>>1, count, n)
            }
        }
        return count
    }
    //MARK:--------最大子序和
    /*
     给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     示例:
     输入: [-2,1,-3,4,-1,2,1,-5,4],
     输出: 6
     解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
     进阶:
     如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。
     */
    class func maxSubArray(_ nums: [Int]) -> Int {
        /*
         动态规划O(n)
         
         动态规划的是首先对数组进行遍历，当前最大连续子序列和为 sum，结果为 ans
         如果 sum > 0，则说明 sum 对结果有增益效果，则 sum 保留并加上当前遍历数字
         如果 sum <= 0，则说明 sum 对结果无增益效果，需要舍弃，则 sum 直接更新为当前遍历数字
         每次比较 sum 和 ans的大小，将最大值置为ans，遍历结束返回结果
         */
        var nums = nums
        var ans = nums[0]
        var sum : Int = 0
        for num in nums {
            if sum > 0{
                sum += num
            }else{
                sum = num
            }
            ans = max(ans, sum)
            
        }
        return ans
    }
    class func maxSubArray1(_ nums: [Int]) -> Int {
        /*
         暴力循环遍历 O(n^2)
         */
        var nums = nums
        //存最大值
        var maxValue = nums[0]
        var sum = 0
        
        for i in 0..<nums.count {
            sum = 0
            for j in i..<nums.count{
                sum += nums[j]
                if sum > maxValue{
                    maxValue = sum
                }
            }
        }
        return maxValue
    }
    class func maxSubArray2(_ nums: [Int]) -> Int {
        /*
         思路：
         1、定义一个max记录过程中最大值
         2、定义lSum、rSum从两头向中间推进的记录的两个最终子序和
         3、到中间汇聚，再取最大值：Math.max(max, lSum+rSum);
         */
        var nums = nums
        //过程中最大值
        var maxValue = max(nums[0], nums[nums.count-1])
        //左半部分，最近一次子序和
        var lSum = 0
        //右半部分，最近一次子序和
        var rSum = 0
        
        var i = 0, j = nums.count-1
        while i<=j {
            lSum = lSum>0 ? lSum+nums[i] : nums[i]
            maxValue = max(maxValue, lSum)
            
            if j != i {
                rSum = rSum>0 ? rSum+nums[j] : nums[j]
                maxValue = max(maxValue, rSum)
            }
            
            i += 1
            j -= 1
        }
        
        //汇聚
        //maxValue 左右两边最大的，lSum+rSum 中间聚合
        return max(maxValue, lSum+rSum)
    }
    class func maxSubArray3(_ nums: [Int]) -> Int {
        /*
         分治法思路：nlog(n)
         通过递归分治不断的缩小规模，问题结果就有三种，左边的解，右边的解，以及中间的解（有位置要求，从中介mid向两边延伸寻求最优解），得到三个解通过比较大小，等到最优解。
         */
        return maxSubArrayPart(nums, 0, nums.count-1)
    }
    private class func maxSubArrayPart(_ nums : [Int], _ left : Int, _ right : Int)->Int{
        if left == right {
            return nums[left]
        }
        
        let mid = (left+right)/2
        return max(maxSubArrayPart(nums, left, mid), max(maxSubArrayPart(nums, mid+1, right), maxSubArrayAll(nums, left, mid, right)))
    }
    //左右两边合起来求解
    private class func maxSubArrayAll(_ nums : [Int], _ left : Int, _ mid : Int, _ right : Int)->Int{
        var leftSum = -2147483648
        var sum = 0
        var i = mid
        while i >= left {
            sum += nums[i]
            if sum > leftSum {
                leftSum = sum
            }
            i -= 1
        }
        sum = 0
        var rightSum = -2147483648
        var j = mid+1
        while j<=right {
            sum += nums[j]
            if sum > rightSum{
                rightSum = sum
            }
            j += 1
        }
        
        return leftSum+rightSum
    }
    //MARK:--------螺旋矩阵
    /*
     给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。
     示例 1:
     输入: [[ 1, 2, 3 ],[ 4, 5, 6 ],[ 7, 8, 9 ]]
     输出: [1,2,3,6,9,8,7,4,5]
     示例 2:
     输入: [[1, 2, 3, 4],[5, 6, 7, 8],[9,10,11,12]]
     输出: [1,2,3,4,8,12,11,10,9,5,6,7]
     */
    class func spiralOrder(_ matrix: [[Int]]) -> [Int] {
        /*
         方法 1：模拟O(n)
         
         直觉
         
         绘制螺旋轨迹路径，我们发现当路径超出界限或者进入之前访问过的单元格时，会顺时针旋转方向。
         
         算法
         
         假设数组有R 行C 列，seen[r][c] 表示第 r 行第 c 列的单元格之前已经被访问过了。当前所在位置为(r, c)，前进方向是di。我们希望访问所有R xC 个单元格。
         
         当我们遍历整个矩阵，下一步候选移动位置是(cr, cc)。如果这个候选位置在矩阵范围内并且没有被访问过，那么它将会变成下一步移动的位置；否则，我们将前进方向顺时针旋转之后再计算下一步的移动位置。
         */
        var ans : [Int] = [Int]()
        if matrix.count == 0 {
            return ans
        }
        let R = matrix.count
        let C = matrix[0].count
        var seen : [[Bool]] = [[Bool]].init(repeating: [Bool].init(repeating: false, count: C), count: R)
        
        var dr = [0, 1, 0, -1]
        var dc = [1, 0, -1, 0]
        
        var r = 0, c = 0, di = 0
        
        for i in 0..<R*C {
            ans.append(matrix[r][c])
            seen[r][c] = true
            let cr = r + dr[di]
            let cc = c+dc[di]
            if 0<=cr && cr<R && 0<=cc && cc<C && !seen[cr][cc]{
                r = cr
                c = cc
            }else{
                di = (di+1)%4
                r += dr[di]
                c += dc[di]
            }
        }
        return ans
    }
    class func spiralOrder2(_ matrix: [[Int]]) -> [Int] {
        /*
         方法 2：按层模拟O(n)
         
         直觉
         
         答案是最外层所有元素按照顺时针顺序输出，其次是次外层，以此类推。
         
         算法
         
         我们定义矩阵的第 k 层是到最近边界距离为 k 的所有顶点。例如，下图矩阵最外层元素都是第 1 层，次外层元素都是第 2 层，然后是第 3 层的。
         对于每层，我们从左上方开始以顺时针的顺序遍历所有元素，假设当前层左上角坐标是(r1, c1)，右下角坐标是(r2, c2)。
         
         首先，遍历上方的所有元素(r1, c)，按照c = c1,...,c2 的顺序。然后遍历右侧的所有元素(r, c2)，按照r = r1+1,...,r2 的顺序。如果这一层有四条边（也就是r1 < r2 并且c1 < c2 ），我们以下图所示的方式遍历下方的元素和左侧的元素。
         
         */
        var ans : [Int] = [Int]()
        if matrix.count == 0 {
            return ans
        }
        var r1 = 0, r2 = matrix.count-1
        var c1 = 0, c2 = matrix[0].count-1
        
        while r1<=r2 && c1<=c2 {
            //第一行的数 top
            for c in c1...c2 {
                ans.append(matrix[r1][c])
            }
            //最右边除了第一行的所有最后一个数 right
            var r = r1+1
            while r <= r2 {
                ans.append(matrix[r][c2])
                r += 1
            }
            //下边及左边最外层数
            if r1<r2 && c1<c2 {
                //bottom
                var c = c2-1
                while c > c1 {
                    ans.append(matrix[r2][c])
                    c -= 1
                }
                //left
                var r = r2
                while r>r1{
                    ans.append(matrix[r][c1])
                    r -= 1
                }
            }
            r1 += 1
            r2 -= 1
            c1 += 1
            c2 -= 1
        }
        
        return ans
    }
    
    class func spiralOrder3(_ matrix: [[Int]]) -> [Int] {
        /*
         方法 3：从外部向内部逐层遍历打印矩阵，最外面一圈打印完，里面仍然是一个矩阵
         第i层矩阵的打印，需要经历4个循环
         从左到右
         从上倒下
         从右往左，如果这一层只有1行，那么第一个循环已经将该行打印了，这里就不需要打印了，即 （m-1-i ）!= i
         从下往上，如果这一层只有1列，那么第2个循环已经将该列打印了，这里不需要打印，即(n-1-i) != i
         */
        var ans : [Int] = [Int]()
        if matrix.count == 0 {
            return ans
        }
        let m = matrix.count
        let n = matrix[0].count
        var i = 0
        
        //统计矩阵从外向内的层数，如果矩阵为空，那么它的层数至少是1层
        let count = (min(m, n)+1)/2
        //从外部向内部遍历，逐层打印数据
        while i<count {
            //从左到右
            for j in i..<n-i {
                ans.append(matrix[i][j])
            }
            //从上到下
            for j in i+1..<m-i{
                ans.append(matrix[j][(n-1)-i])
            }
            
            //从右往左，如果这一层只有1行，那么第一个循环已经将该行打印了，这里就不需要打印了，即 （m-1-i ）!= i
            var j = (n-1)-(i+1)
            while j >= i && (m-1-i != i){
                ans.append(matrix[m-1-i][j])
                j -= 1
            }
            
            //从下往上，如果这一层只有1列，那么第2个循环已经将该列打印了，这里不需要打印，即(n-1-i) != i
            var k = (m-1)-(i+1)
            while k>=i+1 && (n-1-i != i){
                ans.append(matrix[k][i])
                k -= 1
            }
            
            i += 1
        }
        
        return ans
    }
    //MARK:--------跳跃游戏
    /*
     给定一个非负整数数组，你最初位于数组的第一个位置。
     数组中的每个元素代表你在该位置可以跳跃的最大长度。
     判断你是否能够到达最后一个位置。
     示例 1:
     输入: [2,3,1,1,4]
     输出: true
     解释: 从位置 0 到 1 跳 1 步, 然后跳 3 步到达最后一个位置。
     示例 2:
     输入: [3,2,1,0,4]
     输出: false
     解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
     */
    class func canJump(_ nums: [Int]) -> Bool {
        /*
         回溯：从第一个位置开始，模拟所欲可以跳到的位置，然后从当前位置上重复上述操作，当没有办法继续跳的时候，就回溯
         时间复杂度：O(2^n),最多有2^n种从第一个位置到最后一个位置的跳跃方式，其中n 是数组 nums 的元素个数，
         空间复杂度：O(n)，回溯法只需要栈的额外空间。

         */
        return canJumpFromPosition(0, nums)
    }
    private class func canJumpFromPosition(_ position: Int, _ nums: [Int])->Bool{
        if position == nums.count-1 {
            return true
        }
        let furthestJump = min(position+nums[position], nums.count-1)
        //  循环优化
//        var nextPosition = furthestJump
//        while nextPosition > position{
//            if canJumpFromPosition(nextPosition, nums) {
//                return true
//            }
//            nextPosition -= 1
//        }
        for nextPosition in position+1...furthestJump{
            if canJumpFromPosition(nextPosition, nums) {
                return true
            }
        }
        return false
    }
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
    //MARK:--------
 
    //字符转换为整数值
    class func characterTwoSubInt(_ s1 : String, _ s2: String)->Int{
        var number1 = 0, number2 = 0
        for code in s1.unicodeScalars {
            number1 = Int(code.value)
        }
        for code in s2.unicodeScalars{
            number2 = Int(code.value)
        }
        return number1-number2
    }
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
    
    print("组合总和")
//    print(Array_Algorithm.combinationSum([2, 3, 6, 7], 7))
    print(Array_Algorithm.combinationSum([2, 3, 5], 8))
    print("\n")
    
    print("组合总和II")
    print(Array_Algorithm.combinationSum2([10, 1, 2, 7, 6, 1, 5], 8))
//    print(Array_Algorithm.combinationSum2([2, 5, 2, 1, 2], 8))
    print("\n")
    
    print("缺失的第一个正数")
    print(Array_Algorithm.firstMissingPositive([1, 2, 0]))
    print(Array_Algorithm.firstMissingPositive([3, 4, -1, 1]))
    print(Array_Algorithm.firstMissingPositive([7, 8, 9, 11, 12]))
    print(Array_Algorithm.firstMissingPositive2([1, 2, 0]))
    print(Array_Algorithm.firstMissingPositive2([3, 4, -1, 1]))
    print(Array_Algorithm.firstMissingPositive2([7, 8, 9, 11, 12]))
    print("\n")
    
    print("接雨水")
    print(Array_Algorithm.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(Array_Algorithm.trap2([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(Array_Algorithm.trap3([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(Array_Algorithm.trap4([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print("\n")
    
    print("字符串相乘")
    print(Array_Algorithm.multiply("2", "3"))
    print(Array_Algorithm.multiply("123", "456"))
    print(Array_Algorithm.multiply2("2", "3"))
    print(Array_Algorithm.multiply2("123", "456"))
    print("\n")
    
    print("通配符匹配")
    print(Array_Algorithm.isMatch("aa", "a"))
    print(Array_Algorithm.isMatch("aa", "*"))
    print(Array_Algorithm.isMatch("cb", "?a"))
    print(Array_Algorithm.isMatch("adceb", "*a*b"))
    print(Array_Algorithm.isMatch("acdcb", "a*c?b"))
    print("\n")
    print(Array_Algorithm.isMatch2("aa", "a"))
    print(Array_Algorithm.isMatch2("aa", "*"))
    print(Array_Algorithm.isMatch2("cb", "?a"))
    print(Array_Algorithm.isMatch2("adceb", "*a*b"))
    print(Array_Algorithm.isMatch2("acdcb", "a*c?b"))
    print("\n")
    print(Array_Algorithm.isMatch3("aa", "a"))
    print(Array_Algorithm.isMatch3("aa", "*"))
    print(Array_Algorithm.isMatch3("cb", "?a"))
    print(Array_Algorithm.isMatch3("adceb", "*a*b"))
    print(Array_Algorithm.isMatch3("acdcb", "a*c?b"))
    print("\n")
    
    print("跳跃游戏II")
    print(Array_Algorithm.jump([2, 3, 1, 1, 4]))
    print(Array_Algorithm.jump([2, 3, 1, 1, 4]))
    print("\n")
    
    print("全排列")
    print(Array_Algorithm.permute([1, 2, 3]))
    print("\n")
    
    print("旋转图像")
    var rotateArr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
//    Array_Algorithm.rotate(&rotateArr)
//    Array_Algorithm.rotate2(&rotateArr)
    Array_Algorithm.rotate3(&rotateArr)
    print("\n")
    
    print("字母异位词分组")
    print(Array_Algorithm.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    print(Array_Algorithm.groupAnagrams2(["eat", "tea", "tan", "ate", "nat", "bat"]))
    print(Array_Algorithm.groupAnagrams3(["eat", "tea", "tan", "ate", "nat", "bat"]))
    print("\n")
    
    print("pow(x, n)")
    print(Array_Algorithm.myPow(2.00000, 10))
    print(Array_Algorithm.myPow(2.10000, 3))
    print(Array_Algorithm.myPow(2.00000, -2))
    print(Array_Algorithm.myPow2(2.00000, 10))
    print(Array_Algorithm.myPow2(2.10000, 3))
    print(Array_Algorithm.myPow2(2.00000, -2))
    print(Array_Algorithm.myPow3(2.00000, 10))
    print(Array_Algorithm.myPow3(2.10000, 3))
    print(Array_Algorithm.myPow3(2.00000, -2))
    print(Array_Algorithm.myPow4(2.00000, 10))
    print(Array_Algorithm.myPow4(2.10000, 3))
    print(Array_Algorithm.myPow4(2.00000, -2))
    print("\n")
    
    print("N皇后")
    print(Array_Algorithm.solveNQueens(4))
    print("\n")
    
    print("N皇后II")
    print(Array_Algorithm.totalNQueens(4))
    print(Array_Algorithm.totalNQueens2(4))
    print("\n")
    
    print("最大子序和")
    print(Array_Algorithm.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
    print(Array_Algorithm.maxSubArray1([-2,1,-3,4,-1,2,1,-5,4]))
    print(Array_Algorithm.maxSubArray2([-2,1,-3,4,-1,2,1,-5,4]))
    print(Array_Algorithm.maxSubArray3([-2,1,-3,4,-1,2,1,-5,4]))
    print(Array_Algorithm.maxSubArray3([-2, -1]))
    print("\n")
    
    print("螺旋矩阵")
    print(Array_Algorithm.spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print(Array_Algorithm.spiralOrder2([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print(Array_Algorithm.spiralOrder3([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print("\n")
}
