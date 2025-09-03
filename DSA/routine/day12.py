
# radix sort implementation

my_arry = [170, 45, 75, 90, 802, 24, 2, 66]
print("My origina array :" , my_arry)
radixArray = [[], [], [], [], [], [], [], [], [], []]
max_arry = max(my_arry)
exp = 1

while max_arry // exp > 0 :

    while len(my_arry) > 0 :
        val = my_arry.pop()
        radixIndex = (val // exp) % 10
        radixArray[radixIndex].append(val)
    
    for bucket in radixArray :
        while len(bucket) > 0 :
            val = bucket.pop()
            my_arry.append(val)
    
    exp *=10

print("sorted array:",my_arry)

#repeating the radix sort 

myArray = [170, 45, 75, 90, 802, 24, 2, 66]
max_val = max(myArray)
exp = 1

while max_val // exp > 0 :

    while len(myArray) > 0:
        val = myArray.pop()
        radixIndex = (val // exp) % 10
        radixArray[radixIndex].append(val)
    for bucket in myArray:
        while len(bucket) > 0:
            val = bucket.pop()
            myArray.append(val)
    
    exp *= 10
    