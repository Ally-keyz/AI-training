
# revision on selection sort
# what i remember is that it finds the lowest item in the array and put it on the top position

my_array = [14, 8, 3, 1 , 9 , 10 , 11 ]
n = len(my_array)

for i in range(n - 1):
    min_index = i
    for j in range(i + 1 , n):
        if my_array[j] < my_array[min_index]:
            min_index = j
    min_val = my_array.pop(min_index)
    my_array.insert(i, min_val)

print("Sorted:",my_array)

# improved selection sort
my_array2 = [15 , 14 , 2 , 0 , 5 , 8 , 9]
n2 = len(my_array2)

for i in range(n2):
    min_index = i
    for j in range(i + 1 , n):
        if my_array2[j] < my_array2[min_index] :
            min_index = j
    my_array2[i] , my_array2[min_index] = my_array2[min_index] , my_array2[i]

print('Improved algoritm:',my_array2)

# revision on buble sort
my_array3 = [15 , 14 , 2 , 0 , 5 , 8 , 9]
n4 = len(my_array3)

for i in range(n4 -1):
    for j in range(n4 - i - 1):
        if my_array3[j] > my_array3[j+1]:
            my_array3[j] , my_array3[j+1] = my_array3[j+1] , my_array3[j]

print("Buble sorted:",my_array3)

# working on insertion sort
array = [64, 34, 25, 12, 22, 11, 90, 5]
n = len(array)
for i in range(1,n):
    insert_index = i
    current_value =array.pop(i)
    for j in range(i-1 , -1 , -1):
        if array[j] > current_value:
            insert_index = j
    array.insert(insert_index , current_value)


print("Insertion sort:",array)
