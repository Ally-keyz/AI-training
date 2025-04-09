
# we are going to calculate the maximum number in an array using 0(n) complexity

def complexity01(array):
    if len(array) <= 1:
        return array
    
    max_num = array[0]
    #loop through the array to find the number
    for item in array:
        if item > max_num:
            max_num = item
    return max_num


print(complexity01([1,4,5,3,2]))

#finding the minimum element in a given array

def complexity001(nums):
    if len(nums) <= 1:
        return nums
    
    #declare the initial min
    min_num = nums[0]
    for item in nums:
        if item < min_num:
            min_num = item

    return min_num      

#print(complexity001([2,6,3,7,8]))  
#print(complexity001([4, 4, 4]))
#print(complexity001([-1, -99, -5])) 
#print(complexity001([]))  

#check if the list contains duplicates using O(n)

def check(nums):
    if len(nums) < 1:
        return None
    
    #lets loop the numbers
    items = set()
    for item in nums:
        if item in items:
            return item
        else:
            items.add(item)

print(check([1, 2, 3, 4, 2]))

#cheking if the list contains duplicates usin O(n2)

def Check01(nums):
    if len(nums) < 1:
        return None 
    #loop through the entire array
    for item in range(len(nums)):
        for num in range(item+1,len(nums)):
            if nums[item] == nums[num]:
                return nums[item]

print(Check01([1, 2, 3, 4, 2]))      

#cheking if the array contains duplicates and return the exact duplicate and if none return none

def Check02(nums):
    if len(nums) < 1:
        return None
    
    #loop through the entire array
    for item in range(len(nums)):
        for num in range(item + 1 , len(nums)):
            if nums[item] == nums[num]:
                return nums[item]

    return None

# me solving the classic two sum using the O(n2) complexity

def two_sum(nums,target):
    if len(nums) < 1:
        return None
    for item in range(len(nums)):
        for num in range(item + 1 , len(nums)):
            if nums[item] + nums[num] == target:
                return nums[item] , nums[num]

    return None

print(two_sum([2,7,11,15],9))           

# me solving the classic two sum using O(n)
def two_sum01(nums,target):
    if len(nums) < 1:
        return None
    
    indexes = {}
    for index , num in enumerate(nums):
        complement = target - num 
        if complement in indexes:
            return nums[indexes[complement]], num
        indexes[num] = index

print(two_sum01([2,7,11,15],9))   

# usin O(n2) complexity to return all the posible pairs that would add up to the target
def pair_up(nums,target):
    if len(nums) < 1:
        return None
    pairs = []
    for item in range(len(nums)):
        for num in range(item + 1 , len(nums)):
            if nums[item] + nums[num] == target:
                if (nums[item] ,nums[num])  in pairs or (nums[num] , nums[item]) in pairs:
                    continue
                else:
                    pairs.append((nums[item],nums[num])) 

    return pairs


print(pair_up([1, 2, 3, 2, 4, 3], 5))


#Home work
# 1. Count Frequencies of Items (O(n)) 
# 2.  Two Sum — Optimized (O(n))
# 3. Return the First Duplicate Value (O(n))
# 4. Is Anagram? (O(n))
#🔤 Write a function that checks if two strings are anagrams (same letters, different order).
# 5. Find Missing Number from 1 to N (O(n))
#🔎 Given a list with numbers from 1 to n, but with one number missing, return the missing number.
        
    