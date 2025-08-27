# using my knowledge to solve the classic two sum

def twoSum(nums , target):
    answer = []
    for i in range(len(nums)):
        for j in range(i+1 , len(nums)):
            if nums[j] + nums[i] == target:
                answer.append((nums[j] , nums[i]))
    return answer

print(twoSum([8,2,10,11,3,5 ,4],7))

