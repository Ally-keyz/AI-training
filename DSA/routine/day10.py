#implementing counting sort
def countingSort(arry):
    max_val = max(arry)
    count = [0] * (max_val + 1)

    while len(arry) > 0:
        num = arry.pop(0)
        count[num] += 1

    for i in range(len(count)) :
        while count[i] > 0 :
            arry.append(i)
            count[i] -= 1

    return arry

unsortedArr = [4, 2, 2, 6, 3, 3, 1, 6, 5, 2, 3]
sortedArr = countingSort(unsortedArr)

print("Sorted : ", sortedArr)

# using quick sort for the same problem

def partition(arry , low  , high):
    pivot = arry[high]
    i = low - 1

    for j in range(low , high):
        if arry[j] < pivot :
            i += 1
            arry[i] , arry[j] = arry[j] , arry[i]
    arry[i + 1] , arry[high] = arry[high] , arry[i + 1]
    return i+ 1

def quickSort(arry , low  = 0, high = None):
    if high is None :
        high = len(arry) - 1

    if low < high :
        pivot = partition(arry , low  , high)
        quickSort(arry , low , pivot - 1)
        quickSort(arry , pivot + 1 , high)


my_arry = [4, 2, 2, 6, 3, 3, 1, 6, 5, 2, 3]
quickSort(my_arry)
print("Quick sort :" , my_arry)


# revision on implementing insertion sort 
# insertion sort divides the array into two parts the sortesd part and the unsorted part

arrray = [4, 2, 2, 6, 3, 3, 1, 6, 5, 2, 3]
n = len(arrray)

for i in range(1 , n):
    insert_index = i
    current_values = arrray.pop(i)
    for j in range(i-1 , -1 , -1):
        if arrray[j] > current_values :
            insert_index = j
    arrray.insert(insert_index , current_values)

print("Used insertion sort :" , arrray)


# revising on the selection sort
# selection sort looks for the smallest value in the array and then
#  pops it out and then pushes it on the top where it belongs

arrray = [4, 2, 2, 6, 3, 3, 1, 6, 5, 2, 3]
n   = len(arrray)

for i in range(n - 1) :
    min_index = i
    for j in range(i + 1 , n):
        if arrray[j] < arrray[min_index]:
            min_index = j
    min_value = arrray.pop(min_index)
    arrray.insert(i , min_value)

print("Used selection sort :", arrray)


# looking onto buble sort
arrray = [4, 2, 6, 3, 1, 5]
n   = len(arrray)

for i in range(n-1):
    for j in range(n - i -1 ):
        if arrray[j] > arrray[j+ 1] :
            arrray[j] , arrray[j+ 1] = arrray[j+ 1] , arrray[j]

print("Buble sort:", arrray)



