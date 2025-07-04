
# we are going to create a program that calculates the frequecy of an item in an array
def main(array):
    if len(array) <= 1 :
        return array
    
    nums = set()
    freq = {}
    for item in array:
        if item in nums:
            freq[item] += 1
        else:
            nums.add(item)
            freq[item] = 1

    return freq

print(main([1,1,1,2,2,2,2,3,3,4,4,5,6,7]))            
