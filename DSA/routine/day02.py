
# writting an algorithm that finds the lowest item in an array

my_array = [7 , 8 , 4, 20 , 16 , 17 , 3 , 9 , 8 , 11 , 10 , 1 , 0 , -1 , 2 , 50]
minVal = my_array[0]

def search():
    # we are going to loop through all the elements to find the lowest values in these elements
    global minVal
    for num in my_array:
        if num < minVal :
            minVal = num

search()
print("The minimum value is:",minVal)

# implementing buble sort
my_array = [64, 34, 25, 12, 22, 11, 90, 5]
n = len(my_array)

for i in range(n - 1):
    for j in range(n - i - 1):
        if my_array[j] > my_array[j + 1]:
            my_array[j] , my_array[j + 1] = my_array[j + 1] , my_array[j]



# bubble sort improvement 

for i in range(n - 1):
    swaped = False
    for j in range(n - i - 1):
        if my_array[j] > my_array[j + 1] :
            my_array[j] , my_array[j + 1] = my_array[j + 1] , my_array[j]
            swaped = True
        if not swaped :
            break
print('Sorted array:',my_array)