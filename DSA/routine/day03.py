
# revision on buble sort

my_array = [8 ,5 , 12 , 9 , 10 ,6 , 3]
array_len = len(my_array)

for i in range(array_len - 1):
    for j in range (array_len - i - 1):
        if my_array[j] > my_array[j+1] :
            my_array[j] , my_array[j+1] = my_array[j+1] , my_array[j]

print("Sorted array :" , my_array)

# trying to implement selection sort on my own

my_array2 = [64, 34, 25, 5, 22, 11, 90, 12]
n = len(my_array2)

for i in range( n - 1):
    min_index  = i
    for j in range(i + 1 , n):
        if my_array2[j] < my_array2[min_index]:
            min_index = j
    min_value  = my_array2.pop(min_index)
    my_array2.insert(i , min_value)

print("Sorted :", my_array2)

# improved selection sort 

my_array3 = [64, 34, 25, 12, 22, 11, 90, 5]
n = len(my_array3)

for i in range(n):
    min_index = i
    for j in range(i+1,n):
        if my_array3[j] < my_array3[min_index]:
            min_index = j
    my_array3[i] , my_array3[min_index] = my_array3[min_index] , my_array3[i]

print("Sort improved :", my_array3)        