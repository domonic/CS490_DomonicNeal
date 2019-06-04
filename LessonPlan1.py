'''Input the string “Python” as a list of characters from console, delete at least 2 characters, reverse the resultant
string and print it.'''

'''variables'''
user_string = input("Please enter a string: ")
string_list = []
result_string = " "


'''loop through the input string from the user and add each letter to a list'''
for char in user_string:
    string_list.append(char)

'''delete 2 of the characters from the string'''
del string_list[1:3]

'''take the characters from the string_list variable and turn them back into a string'''
result_string = ''.join(map(str, string_list))

'''reverse string'''
print((result_string[::-1]))

print()
print()


'''Take two numbers from user and perform arithmetic operations on them.'''

'''get user integer input'''
user_num1 = int(input("Please enter an integer: "))
user_num2 = int(input("Please enter another integer: "))

'''operation performance on user integers'''
summation = user_num1 + user_num2
difference = user_num1 - user_num2
product = user_num1 * user_num2
division = user_num1 / user_num2

'''output results'''
print(summation)
print(difference)
print(product)
print(division)

print()
print()


''' Write a program that accepts a sentence and replace each occurrence of ‘python’ with ‘pythons’ without using 
regex'''

'''variables'''
pythons_string = input("Please enter a string:")


'''iterate through string to find occurrences of the word python and replace with pythons'''
new_pythons_string = pythons_string.replace("python", "pythons")

'''print new resultant string'''
print("Result String: ", new_pythons_string)























