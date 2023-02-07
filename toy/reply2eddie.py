# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 3/2/2023 12:28 pm

"""


# names = ["eddie", "jasper", "alice"]
# names[-1] = names[-1].upper()
# print(names)

# l = [0, 1, '2', 3, 4, '4', 5, 7, 6, 9, 1]
# print(int(l[2]) + l[-1])
# l1 = l[:6]
# l2 = l[6:]
# print(l1)
# print(l2)

# Age={'Bo':32, 'Lani':27, 'Manu':35}
# Age['Lani'] = 18
# print(Age)

StuDict = {"mars":123,"ivy":456,"summer":789,"helen":101}
student_name = input("Please input student name:")
if student_name in StuDict:
    print(StuDict[student_name])
else:
    print("%s is not in this class" % student_name)

# tgtList = []
# with open("data.txt", "r") as fr:
#     for line in fr:
#         if "CBS" in line:
#             tgtList.append(line)
# for line in tgtList:
#     print(line)
# with open("save.txt", "w") as fw:
#     fw.writelines(tgtList)




