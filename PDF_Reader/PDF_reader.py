import pdfplumber


class PDF_reader:
    def __init__(self, font_list: list):
        self.font_list = font_list

    def Read(self, fileRoute: str, outputFileRoute: str):
        str_all = ""
        with pdfplumber.open(fileRoute) as pdf:
            for page_i in pdf.pages:
                for i in page_i.chars:
                    if len(i['text']):
                        flag = 0
                        for font in self.font_list:
                            if font in i['fontname']:
                                flag = 1
                        if flag:
                            str_all += i['text']

        str_all = str_all.split(" ")
        str_res = []
        for i in str_all:
            if len(i) > 0:
                str_res.append(i)
        with open(outputFileRoute, 'w', encoding='utf-8') as f:
            for i in str_res:
                f.write(i + '\n')


class PDF_reader1_1:
    def __init__(self, key_word_list: list):
        self.key_word_list = key_word_list
        self.str_res = {}
        for i in self.key_word_list:
            self.str_res[i] = []

    @staticmethod
    def findAnchor(pos, StrInput: str):
        p = pos + 1
        while p < len(StrInput):
            p1 = StrInput.find("#", p)
            if p1 == -1:
                return -1
            if p1+2 < len(StrInput) and StrInput[p1+2] == "#":
                return p1
            else:
                p = p1+2
        return -1

    def mapKeyWord(self, strInput: str):
        cnt = 0
        for i in self.key_word_list:
            strInput = strInput.replace(i, '#' + str(cnt) + '#')
            cnt += 1
        return strInput

    def Split(self, strInput):
        strNew = self.mapKeyWord(strInput)
        tab = ["None" for _ in range(len(strNew) + 10)]
        p = 1
        while p < len(strNew) and p != -1:
            p1 = self.findAnchor(p, strNew)
            if p1 == -1:
                break
            p2 = self.findAnchor(p1, strNew)
            if p1 != -1 and p2 == -1:
                p2 = len(strNew)
            for i in range(p1+3, p2, 1):
                tab[i] = strNew[p1 + 1]
            p = p2-1
        flag = False
        for i in range(len(strNew)):
            if tab[i] == 'None' and tab[i+1] != 'None':
                flag = True
                self.str_res[self.key_word_list[int(tab[i+1])]].append(["", i])
            if tab[i-1] != 'None' and tab[i] == 'None':
                flag = False
            if tab[i] == 'None':
                continue
            tab_i = tab[i]
            belongs = self.key_word_list[int(tab_i)]
            str_i = strNew[i]
            if flag:
                self.str_res[belongs][-1][0] += str_i

    def Read(self, fileRoute: str, outputFileRoute: str):
        str_all = ""
        with pdfplumber.open(fileRoute) as pdf:
            for page_i in pdf.pages:
                for i in page_i.chars:
                    if len(i['text']):
                        str_all += i['text']
        self.Split(str_all)
        sortList = []
        for Type in self.str_res:
            for i in self.str_res[Type]:
                k = i
                k[0] = Type + k[0]
                sortList.append(k)
        sortList = sorted(sortList, key=lambda x: x[1])
        with open(outputFileRoute, 'w', encoding='utf-8') as f:
            for data in sortList:
                f.write(str(data[0]+"\n"))


PDF_reader1_1(["问：", "答："]).Read("样例1.pdf", "output.txt")
print("end")
