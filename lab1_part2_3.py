<<<<<<< HEAD
# Part 2 Lab





def get_all_substrings(string):

    substrings = []

    length = len(string)

    for i in range(length):

        for j in range(i, length):

            substrings.append(string[i: j+1])

    return substrings



def get_longest(string):

    substrings = get_all_substrings(string)

    longest = ''

    for sub in substrings:

        if len(sub) > len(longest) and no_repeating_chars(sub):

            longest = sub

    return (longest, len(longest))



def no_repeating_chars(string):

    for i in range(len(string)):

        for j in range(len(string)):

            if i == j:

                continue

            elif string[i] == string[j]:

                return False

    return True



# x = input()

x = "pwwkew"



print(get_longest(x))





# Part 3 - Library management system





class Person(object):

    def __init__(self, fname='', lname='', id=0):

        self.fname = fname

        self.lname = lname

        self.id = id





class Student(Person):

    def __init__(self, fname, lname, id, fines_due=0):

        super().__init__(fname, lname, id)

        self.__fines_due = fines_due

        self.checked_out = []



    def check_out_book(self, book):

        self.checked_out.append(book)

        book.check_out()



    def get_fines(self):

        return self.__fines_due





class Book(object):

    def __init__(self, title="", author="", genre="", checked_out=False, pages=0):

        self.title = title

        self.author = author

        self.genre = genre

        self.checked_out = checked_out

        self.pages = pages



    def check_out(self):

        self.checked_out = True



    def check_in(self):

        self.checked_out = False





if __name__ == '__main__':

    student = Student(fname="Joe", lname="Smith", id=5050)

    book = Book("Of mice and men", "John Steinbeck", "Literary Fiction", pages=125)

    student.check_out_book(book)

    print(book.checked_out)

print(4)
=======
# Part 2 Lab





def get_all_substrings(string):

    substrings = []

    length = len(string)

    for i in range(length):

        for j in range(i, length):

            substrings.append(string[i: j+1])

    return substrings



def get_longest(string):

    substrings = get_all_substrings(string)

    longest = ''

    for sub in substrings:

        if len(sub) > len(longest) and no_repeating_chars(sub):

            longest = sub

    return (longest, len(longest))



def no_repeating_chars(string):

    for i in range(len(string)):

        for j in range(len(string)):

            if i == j:

                continue

            elif string[i] == string[j]:

                return False

    return True



# x = input()

x = "pwwkew"



print(get_longest(x))





# Part 3 - Library management system





class Person(object):

    def __init__(self, fname='', lname='', id=0):

        self.fname = fname

        self.lname = lname

        self.id = id





class Student(Person):

    def __init__(self, fname, lname, id, fines_due=0):

        super().__init__(fname, lname, id)

        self.__fines_due = fines_due

        self.checked_out = []



    def check_out_book(self, book):

        self.checked_out.append(book)

        book.check_out()



    def get_fines(self):

        return self.__fines_due





class Book(object):

    def __init__(self, title="", author="", genre="", checked_out=False, pages=0):

        self.title = title

        self.author = author

        self.genre = genre

        self.checked_out = checked_out

        self.pages = pages



    def check_out(self):

        self.checked_out = True



    def check_in(self):

        self.checked_out = False





if __name__ == '__main__':

    student = Student(fname="Joe", lname="Smith", id=5050)

    book = Book("Of mice and men", "John Steinbeck", "Literary Fiction", pages=125)

    student.check_out_book(book)

    print(book.checked_out)

print(4)
>>>>>>> d0a6132cf5e59ae6d1c75f24538aa712897cfe64
