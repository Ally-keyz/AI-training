
# revision of counting sort

def countingSort(arry):
    max_val = max(arry)
    count  = [0] * (max_val + 1)

    while len(arry) > 0 :
        num = arry.pop(0)
        count[num] += 1


    for j in range(len(count)) :
        if count[j] > 0 : 
            arry.append(j)
            count[j] -= 1   


my_arry = [2, 6, 3, 6 , 8, 1]
countingSort(my_arry)
print(my_arry)


# praticing  quick sort

def partitition(arry , low  , high):
    pivot = arry[high]
    i = low - 1

    for j in range(low , high):
        if arry[j] > pivot :
            i += 1
            arry[i] , arry[j] = arry[j] , arry[i]
    arry[i+1] , arry[high] = arry[high] , arry[i+1]
    return i + 1

def quickSort(arry , low=0 , high =None):
    if high is None:
        high = len(arry) - 1

    if low < high :
        pivot = partitition(arry , low , high)
        quickSort(arry , low , pivot -1)
        quickSort(arry , pivot+ 1 , high)       

array = [2, 6, 3, 6 , 8, 1]
quickSort(array)
print("Quick sorted :" , array)