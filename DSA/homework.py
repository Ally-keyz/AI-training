#Home work
# 1. Count Frequencies of Items (O(n)) 
# 2. Two Sum — Optimized (O(n))
# 3. Return the First Duplicate Value (O(n))
# 4. Is Anagram? (O(n))
# Write a function that checks if two strings are anagrams (same letters, different order).
# 5. Find Missing Number from 1 to N (O(n))
#🔎 Given a list with numbers from 1 to n, but with one number missing, return the missing number.



# 1 count frequencies of items in an array O(n) complexity

def counter(array):
    if len(array) < 1:
        return None
    
    nums = set()
    freq = {}
    for item in array:
        if item in nums:
            freq[item] +=1
        else:
            nums.add(item)
            freq[item] = 1

    return freq

print(counter([1,2,2,3,4,5,5]))     


# 2.  Two Sum — Optimized (O(n))
def two_sum(arry,target):
    if len(arry) < 1:
        return None
    
    indexes = {}
    for index , item in enumerate(arry):
        complement = target - item
        if complement in indexes:
            return arry[indexes[complement]] , item
        indexes[item] = index

print(two_sum([2,7,11,15],9))        

#calculate all possibilities to the target using O(n2)

def two_sum01(arry,target):
    if len(arry) < 1:
        return None
    
    sums = set()
    for item in range(len(arry)):
        for num in range(item + 1, len(arry)):
            if arry[item] + arry[num] == target:
                sums.add((arry[item],arry[num]))

    return sums

print(two_sum01([1,2,8,7,4,5],9))            

#using the O(n) complexity to find all the possible cominations to the target
def twoSum_optmised(nums,target):
    if len(nums) < 1:
        return None
    indexes ={}
    sums = set()
    for index , item in enumerate(nums):
        complement = target - item
        if complement in indexes:
            sums.add((nums[indexes[complement]],item))
        else:
            indexes[item] = index

    return sums

print(twoSum_optmised([1,2,8,7,4,5],9))           


#execising the two sum optimized for better and deep understsanding

def two_sumOP(nums,target):
    if len(nums) < 1:
        return None
    indexes = {}
    sum_nums = set()
    for index , num in enumerate(nums):
        complement = target - num
        if complement in indexes:
            sum_nums.add(nums[indexes[complement]],num)
        else:
            indexes[num] = index

#returning the first duplicate value
def duplicates(nums):
    if len(nums) < 1:
        return None
    duplicate = set()
    for num in nums:
        if num in duplicate:
            return num
        duplicate.add(num)

#counting the frequency of a number
def counter_freq(nums):
    if len(nums) > 1:
        return None
    freq = {}
    list = set()
    for num in nums:
        if num in list:
            freq +=1
        else: 
            list.add(num)
            freq = 1

#check if a string are anaragm

def anargm(str1,str2):
    if len(str1) and len(str2) < 1:
        return None
    
    if sorted(str1) == sorted(str2):
        result  = "They are anagrams"
        return result
    else:
        result = "They are not anagrams"
        return result


print(anargm("hello","olleh"))  


#find the missing value from an array of intered numbers from 1 to n

def check(nums):
    if len(nums) < 1:
        return None
    #sort the array for more accuracy
    array_sorted  = sorted(nums)
    values = set()
    missing_num = set()

    for i in range(1,array_sorted[-1]):
        if i not in  array_sorted:
            missing_num.add(i)
        else:
            values.add(i)

    return missing_num        


print(check([1,3,5,6,7]))





    
    

        

              



                
        

            

