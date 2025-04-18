
#challenge from leet code 

def main(arry,k):
    if len(arry) < 1:
        return None
    
    count = 0
    for item in range(len(arry)):
        item_count = 0
        for num in range(item,len(arry)):
            item_count += arry[num]
            if item_count == k:
                count +=1

    return count

# we are going to find the lenght of the longest sub array without duplicate characters

def check(str):
    if len(str) < 1:
        return None
    seen = set()
    duplicated = set()
    for char in str:
        if char in seen:
            seen.remove(char)
        else:
            seen.add(char)

    return seen

print(check("pwwkew"))            

def longest_unique_substring(s):
    left = 0
    seen = set()
    max_length = 0

    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length


print(longest_unique_substring("pwwkew"))



def check2(s):
    if len(s) < 1:
        return None
    
    left = 0
    seen = set()
    max_length = 0

    for char in range(len(s)):
        while s[char] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[char])
        max_length = max(max_length , char - left + 1)

    return max_length    
print(check2("pwwkew")) 



            
    
