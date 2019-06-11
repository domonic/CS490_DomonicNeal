
class Employee:

    employee_count = 0
    total_salary = 0

    '''Constructor to initialize name, department, salary, family'''
    def __init__(self, name, department, salary, family):
        self.name = name
        self.department = department
        self.salary = salary
        self.family = family
        Employee.employee_count += 1
        Employee.total_salary += salary

    '''Output to screen employee name'''
    def get_name(self):
        print("Employee Name: ", self.name)

    '''Output to screen employee department'''
    def get_department(self):
        print("Employee Department: ", self.department)

    '''Output to screen employee salary'''
    def get_salary(self):
        print("Employee Salary: $%s " % self.salary)

    '''Output to screen employee family'''
    def get_family(self):
        print("Employee Family: ", self.family)

    '''Output to screen employee information'''
    def employee_info(self):
        print("Employee Name: ", self.name, " \n"  "Employee Department: ", self.department, "\n" 
              "Employee Salary: $%s" % self.salary, "\n" "Employee Family: ", self.family)

    '''function to output the count of employees'''
    def get_employee_count():
        print("There are a total of %d employees." % Employee.employee_count)

    '''function to calculate the average salary'''
    def avg_salary():
        avg_salary = round((Employee.total_salary / Employee.employee_count), 2)
        print("The average salary for employees is: $%s" % avg_salary)

'''FullTimeEmployee Class full inheritence of Employee Class '''
class FullTimeEmployee(Employee):

    def __init__(self, name, department, salary, family):
        Employee.__init__(self, name, department, salary, family)


'''Employee object instances'''
employee_one = FullTimeEmployee("Artez", "Health Science", 83567, "Johnson")
employee_two = Employee("Cameron", "Biology", 121595, "Johnson")
employee_three = FullTimeEmployee("Domonic", "Information Technology", 253300, "Neal")
employee_four = Employee("Ishmael", "Business Entrepreneurship", 234344, "Shumate")
employee_five = Employee("Aaron", "Biology", 115255, "Walker")
employee_six = FullTimeEmployee("Brandon", "Health Science", 352676, "Woods")
employee_seven = Employee("TraVaughn", "Business Entrepreneurship", 220777, "Watson")

'''Member function get_name'''
employee_two.get_name()
employee_three.get_name()
print()

'''Member function get_department'''
employee_two.get_department()
employee_three.get_department()
print()

'''Member function get_salary'''
employee_two.get_salary()
employee_three.get_salary()
print()

'''Member function get_family'''
employee_two.get_family()
employee_three.get_family()
print()

'''Member function employee_info'''
employee_two.employee_info()
print()
employee_three.employee_info()
print()

'''Total number of employees and their average salaries'''
Employee.get_employee_count()
Employee.avg_salary()


