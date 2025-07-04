# challenge is to remove duplicates from an array

def challenge(arry):
    if len(arry) < 1:
        return arry

    nums = []
    for num in arry:
        if num not in nums:
            nums.append(num)

    return nums     

    


print(challenge([0,0,1,1,1,2,2,3,3,4]))


