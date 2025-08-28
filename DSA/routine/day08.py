
#revision of  insertion sort 
# how it works
# it divides the array into two sections  the sectiion of the sorted values
#  and the section of the un sorted after it takes one element at time from the unsorted part 
#  and compares it with the element in the sorted part and then sorts them


my_array = [2,5,3,8,9,11,1]
n = len(my_array)

#for i in range(1 , n):
   # insert_index = i
   # current_values = my_array.pop(i)
  #  for j in range(i-1 , -1 , -1):
        #if my_array[j] > current_values :
           # insert_index = j
    #my_array.insert(insert_index , current_values)

print(my_array)

# quick sort implementation

def partition( arry , low , high):
    pivot = arry[high]
    i = low - 1
    for j in range(low , high):
        if arry[j] <= pivot:
            i += 1
            arry[i] , arry[j] = arry[j] , arry[i]
    arry[i + 1] , arry[high] = arry[high] , arry[i+1]
    return i + 1

def QuickSort ( arry , low=0 , high = None):
    if high is None:
        high = len(arry) - 1
    if low < high :
        pivot_index = partition(arry , low , high)
        QuickSort(arry , low , pivot_index - 1)
        QuickSort(arry , pivot_index + 1 , high)

my_array = [10, 80, 30, 90, 40, 50, 70, 40, 80]
QuickSort(my_array)
print("sorted :", my_array)

# finding the k-th smallest element using quick sort

def partition ( arry  , low  , high):
    pivot = arry[high]
    i = low - 1
    for j in range(low , high):
        if arry[j] <= pivot:
            i += 1
            arry[i] , arry[j] = arry[j] , arry [i]
    arry[i+ 1] , arry[high] = arry[high] , arry[i+1]
    return i + 1

def quickSort ( arry , low=0 , high = None):
    if high is None:
        high = len(arry) - 1
    if low < high :
        pivot_index = partition(arry , low  , high)
        quickSort( arry , low , pivot_index - 1)
        quickSort( arry , pivot_index + 1 , high)

def kth_element(arry , k):
    unique_arry = list(set(arry))
    quickSort(unique_arry)
    if k <= len(unique_arry):
        return unique_arry[k-1]
    else:
        return "No enough unique elements"

arr = [7, 3, 1, 3, 9, 7, 5]
k=3
print("K-th element :",kth_element(arr,k))
        
