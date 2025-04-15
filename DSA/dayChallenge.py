#question : you are given an array of integers and an interger k
# Then find the total number of all sub arrays which could add up to that number
#am going to be using O(n2) time complexity

def check(nums,k):
    if len(nums) < 1:
        return None
    count = 0
    for num in range(len(nums)):
        current_sum = 0
        for sub in range(num,len(nums)):
            current_sum += nums[sub]
            if current_sum == k:
                count += 1

    return count       

print(check([1,1,1],2))

#am gong to try this approach using O(n) time complexity

def check2(nums,k):
    if len(nums) < 1:
        return None
    
    count  = 0
    prefix_sum = 0
    prefix_map = {0:1}

    for num in nums:
        prefix_sum += num

        if (prefix_sum - k) in prefix_map:
            count += prefix_map[prefix_sum - k]

        prefix_map[prefix_sum] = prefix_map.get(prefix_sum , 0) + 1    

    return count

print(check2([-2, -1, 2, 1],1))


# repeat the daily challenge 
def main(nums,target):
    if len(nums) < 1:
        return None
    
    count = 0
    for num in range(len(nums)):
        count_sum = 0
        for item in range(num , len(nums)):
            count_sum += nums[item]

            if count_sum == target:
                count += 1

    return count
            


    




