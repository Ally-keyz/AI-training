
#revision of insertion sort

my_array = [8 , 7  , 3 , 5 , 11 , 1]
n = len(my_array)

for i in range(1,n):
    insert_index = i
    current_value = my_array.pop(i)
    for j in range(i -1 , -1 , -1):
        if my_array[j] > current_value :
            insert_index = j
    my_array.insert(insert_index,current_value)

print(my_array)

# revision on selection sort

array = [8 , 7  , 3 , 5 , 11 , 1]
m = len(array)

for i in range(m-1):
    min_index = i
    for j in range(i+1 , m):
        if array[j] < array[min_index]:
            min_index = j
    min_val = array.pop(min_index)
    print(min_val)
    array.insert(i , min_val)

print(array)

# implementing buble sort
numbers = [4 , 7 , 2 , 3 , 6 , 8 , 1 , 9]
leng = len(numbers)
for i in range(leng - 1):
    min_index = i
    for j in range(i + 1 , leng):
        if numbers[j] < numbers[min_index]:
            numbers[j] , numbers[min_index] = numbers[min_index] ,numbers[j]

print(numbers)

# repeating insertion sort 
my_array2 =  [4 , 7 , 2 , 3 , 6 , 8 , 1 , 9]
n2 = len(my_array2)

for i in range(1 , n2):
    insert_index = i
    current_value = my_array2.pop(i)
    for j in range(i - 1 , -1 , -1):
        if my_array2[j] > current_value :
            insert_index = j
    my_array2.insert(insert_index , current_value)

print(my_array2)


          




 