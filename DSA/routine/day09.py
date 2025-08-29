
# revision of quick sort

def partition(arry , low  , high):
    pivot = arry[high]
    i = low - 1
    for j in range(low , high):
        if arry[j] <= pivot :
            i += 1 
            arry[i] , arry[j] = arry[j] , arry[i]
    arry[i+1] , arry[high] = arry[high] , arry[i+1]
    return i + 1

def quickSort(arry , low=0 , high = None):

    if high is None :
        high = len(arry) - 1

    if low < high :
        pivot_index = partition(arry , low , high)
        quickSort(arry , low , pivot_index - 1)
        quickSort(arry , pivot_index + 1 , high)

my_arry = [7, 3, 1, 3, 9, 7, 5]
quickSort(my_arry)
print(my_arry)

# practicing insertion sort
my_arry = [7, 3, 1, 3, 9, 7, 5]
n = len(my_arry)
for i in range(1 , n):
    insert_index = i
    current_values = my_arry.pop(i)
    for j in range(i-1 , -1 ,-1):
        if my_arry[j] > current_values :
            insert_index = j
    my_arry.insert(insert_index , current_values)

print("Sorted :",my_arry)

# leet code test

# q1 sorting colors quiz

def sortColors(arry):
    n = len(arry)
    for i in range(n-1):
        for j in range(i + 1 , n):
            if arry[j] < arry[i]:
                arry[j] , arry[i] = arry[i] , arry[j]
    return arry

print(sortColors([2,0,2,1,1,0]))            


# q2 finding the kth lagest element

def partition2(arry  , low  , high) :
    pivot  = arry[high]
    i = low - 1
    for j in range(low , high) :
        if arry[j] >= pivot:
            i += 1
            arry[i] , arry[j] = arry[j] , arry[i]
    arry[i+1] , arry[high] = arry[high] , arry[i+1]
    return i+1

def sort(arry , low = 0 , high = None):
    if high is None:
        high  = len(arry) - 1
    if low < high :
        pivot = partition2(arry , low , high)
        sort(arry , low  , pivot - 1) 
        sort(arry , pivot + 1 , high)


def kth_element(arry , k):
    unique_arry = list(set(arry))
    sort(unique_arry)

    # fing the kth largest element
    kth = unique_arry[k - 1]
    return kth


array =[3,2,1,5,6,4]
print("Kth element is :",kth_element(array , 2))    
