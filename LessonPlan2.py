'''string module for easier string manipulation'''
import string


'''Write a program, which reads weights (lbs.) of
N students into a list and convert these weights to kilograms in a separate list'''

def pounds_to_kilos():
    # 2.2 pounds = 1 kilo


    '''Get N number of students from the user'''
    student = 0
    num_students = int(input("Please enter the number of students who are needing conversion: "))

    kilos = 0.00
    kilos_list = []
    pounds_list = []

    '''Use loop iteration to get the pound value for each student based on number of students received from user'''
    while student < num_students:
        student_pounds = int(input("Please enter weight in pounds: "))
        pounds_list.append(student_pounds)
        student += 1

    print()
    print("Students in Pounds")
    print(pounds_list)

    '''For each item in the list that was appended based on user input take that item in this case item is pounds
     and convert it to kilos using proper equation'''
    for pounds in pounds_list:
        kilos = round(pounds / 2.20, 2)
        kilos_list.append(kilos)

    print()
    print("Students in Kilos")
    print(kilos_list)


'''Write a program that returns every other char of a given string starting with first using a function 
named “string_alternative”'''

def string_alternative():

    '''Get user input to later change'''
    user_string = input("Please enter your string: ")

    '''Output to screen the users input but only every other character'''
    print(user_string[::2])


'''Write a python program to find the wordcountin a file for each line and then print the output.
Finally store the output back to the file'''

def wordount():

    '''Variable to hold the word in the file and the number of times the particular word appears'''
    file_words = {}

    '''Open file for reading and writing'''
    file = open("LessonPlan2_example.txt", "r+")

    '''Iterate through each line of the file, count how many times a word appears an output it, and then write to
    the file'''
    for line in file:

        '''remove punctuation that may be considered connected to a word'''
        strip_punctuation = str.maketrans({key: None for key in string.punctuation})
        stripped_line = line.translate(strip_punctuation)

        words = stripped_line.split()

        '''Iterate through each word in each line found within the file, check to see if the word is already found if not
        set the word and its count to 1 and if it is found simply increase the word count'''
        for word in words:

            '''change each word into all lowercase to account for the same word but cased differently'''
            word = word.lower()

            '''check if value found is a word oppose to a number or any other entity'''
            if word is int:
                continue

            count = file_words.get(word, 0)
            count += 1
            file_words[word] = count

    '''Iterate through the dictionary items based on their key value pairs to output to screen as well as write to file'''
    for word, num in file_words.items():

        '''Output to screen the word and its number of occurrences'''
        print(word, ": ", num)

        '''Write to file the words and their number of occurrences'''
        file.write(str(word) + ": " + str(num) + '\n')



'''Pounds to kilos conversion function call'''
pounds_to_kilos()
print()

'''Convert user input to output every other character in their string'''
string_alternative()
print()


'''Allow the user to decide if they would like to perform word count on a file'''
begin_count = 0

'''Continue to loop until begin_count is no longer 0 which can only occur if the user enters 1 to invoke the 
wordcount function which upon completion will set begin_count to 1'''
while begin_count == 0:
    request = input("Please press 1 to perform word count on a file: ")

    '''Only invoke the wordcount function if the user enters 1'''
    if request == "1":
        print()

        '''Count the number of occurrences of a word in a text file'''
        wordount()

        '''Set begin_count to 1 so that the loop will be exited'''
        begin_count = 1

















