import os
import traceback
from number_parser import parse_ordinal,parser
import re
import datetime
import math
import logging

logger = logging.getLogger()

from utils import *
from utils import getNumberFromString

month_table = ['January','February','March','April','May','June','July','August','September','October','November','December']
eps = 1e-5
#type_dict = {'TIME':time, 'DURATION':duration, 'PERCENT':percentage, "ORDINAL":ordinal, 'DATE':myDate, 'MONEY': money, 'NUMBER':number }
#
def getNumber(entity):
    if isinstance(entity,int) or isinstance(entity,float) or isinstance(entity,str):
        return number(value=float(entity), original_text=str(entity),version=0)

    text = entity['text']
    if 'normalizedNER' in entity:
        normalizedNER = entity['normalizedNER']
    else:
        normalizedNER = ""
#number
    if entity['ner'] == 'NUMBER':
        if normalizedNER == "":
            #logger.info(f"No normalizedNER: {entity}\nhelp!")
            value = getNumberFromString(text)[0]
        else:
            try:
                text = re.sub(',','',text)
                value = float(text)
            except:
                if 'million' in text.lower() or 'billion' in text.lower() :
                    value = getNumberFromString(normalizedNER)[0]
                else :
                    value = getNumberFromString(normalizedNER)[0]
        return number(value=value, original_text=entity['text'],version=0)

#deltatime or deltadate
    if entity['ner'] == 'DURATION':
        if normalizedNER == "":
            #logger.info(f"No normalizedNER: {entity}\nhelp!")
            return deltaDate([0,0,0],original_text=text, version = 0)
        else :
            normalizedNER = re.sub("OFFSET",'',normalizedNER)
            normalizedNER = normalizedNER.strip()

            if not normalizedNER.startswith('P'):
                value = 0
                unit = ""
            normalizedNER = normalizedNER[1:]

            tord = 0
            if 'X' in normalizedNER:
                normalizedNER = re.sub('X','0',normalizedNER)
            try:
                #handle time duration
                if normalizedNER.startswith('T'):
                    normalizedNER = normalizedNER[1:]
                    #get the time unit: H, M, S...
                    unit = normalizedNER[-1]
                    value = float(normalizedNER[:-1])
                    tord = 1
                else :# handle long time duration: Y,M,W,D
                    unit = normalizedNER[-1]
                    value = float(normalizedNER[:-1])
            except:
                unit = ""
                value = 0
            if unit in ['H','M','S'] and (tord == 1):
                inx = ['H','M','S'].index(unit)
                v = [0,0,0,]
                v[inx] = value
                return deltaTime(v,original_text=text, version=0)
            elif unit in ['Y','M',"D"]:
                inx = ['Y','M',"D"].index(unit)
                v = [0,0,0,]
                v[inx] = value
                return deltaDate(v,original_text=text, version=0)
            else:
                return deltaDate([0,0,0],original_text=text, version = 0)
            
#percent
    if entity['ner'] == 'PERCENT':
        if normalizedNER=="":#
            #logger.info(f"No normalizedNER: {entity}\nhelp!")
            value = getNumberFromString(text)[0]
        else :#
            value = getNumberFromString(normalizedNER)[0]
        return percentage(value=value, original_text=text,version=0)

#ordinal
    if entity['ner'] == 'ORDINAL':
        if normalizedNER == "":
            text = text.lower()
            text = re.sub('and','',text)
            text = re.sub(' ','',text)
            try:
                value = float(text)
            except:
                try:
                    value = float(parse_ordinal(text))
                except :
                    value = getNumberFromString(text)[0]
        else :
            try:
                value = float(normalizedNER)
            except:
                value = getNumberFromString(text)[0]
        return ordinal(value, original_text=text, version = 0)

#mydate
    if entity['ner'] == 'DATE':
        if re.match("[0-9]+ years? from [0-9]{4}",text): #
            number_list = getNumberFromString(text)
            a,b = number_list[0],number_list[1]
            return myDate([round(b),0,0,])
        
        if 'quarter' in entity['text']:
            text = entity['text'].lower()
            List = ['first','second','third','fourth']
            for i in range(4):
                if List[i] in entity['text'].lower():
                    return number(i+1,original_text=entity['text'],version = 0)
            return number(0, original_text=entity['text'],version = 0)


        date = {'year':0, 'month':0, 'day':0}            
        try:#
            date['year'] = int(text)
        except:                

            if normalizedNER == "":
                date = {'year':0, 'month':0, 'day':0}
            else :
                if '/' in normalizedNER:
                    date = {'year':0, 'month':0, 'day':0}
                if "INTERSECT" in normalizedNER:
                    normalizedNER = normalizedNER.split("INTERSECT")[-1].strip()

                if normalizedNER.startswith('OFFSET') or normalizedNER.startswith("THIS") or normalizedNER.startswith("NEXT_IMMEDIATE") or normalizedNER.startswith('P'):
                    normalizedNER = re.sub('OFFSET','',normalizedNER).strip()
                    normalizedNER = re.sub('THIS','',normalizedNER).strip()
                    normalizedNER = re.sub('NEXT_IMMEDIATE','',normalizedNER).strip()
                    if normalizedNER.startswith('P'):
                        normalizedNER = re.sub('P','',normalizedNER).strip()
                        normalizedNER = re.sub('X','1',normalizedNER)
                        if normalizedNER.startswith('T'):
                            normalizedNER = normalizedNER[1:]
                            time_ = [0,0,0 ]
                            try:
                                if normalizedNER[-1] == 'H':
                                    time_[0] = int(normalizedNER[:-1])
                                elif normalizedNER[-1] == 'M':
                                    time_[1] = int(normalizedNER[:-1])
                                elif normalizedNER[-1] == 'S':
                                    time_[2] = int(normalizedNER[:-1])
                            except:
                                pass
                            return deltaTime(time_, original_text= text, version = 0)
                        else:
                            date = {'year':0, 'month':0, 'day':0}
                            try:
                                if normalizedNER[-1] == 'Y':
                                    date = {'year':int(normalizedNER[:-1]), 'month':0, 'day':0}
                                elif normalizedNER[-1] == 'M':
                                    date = {'year': 0, 'month':int(normalizedNER[:-1]), 'day':0}
                                elif normalizedNER[-1] == 'D':
                                    date = {'year': 0, 'month': 0, 'day':int(normalizedNER[:-1])}                                    
                            except:
                                pass
                        return deltaDate(date,original_text = text, version = 0)
                normalizedNER = re.sub("-SU","",normalizedNER)
                normalizedNER = re.sub("-WI","",normalizedNER)
                normalizedNER = re.sub("-FA","",normalizedNER)
                normalizedNER = re.sub("-SP","",normalizedNER)
                try:
                    normalizedNER = int(normalizedNER)
                    date = {'year':normalizedNER, 'month':0, 'day':0}
                except:
                    normalizedNER = normalizedNER.split('-')
                    try:
                        for idx, num in enumerate(normalizedNER):
                            normalizedNER[idx] = int(re.sub('X','0',num))
                        while len(normalizedNER)<3:
                            normalizedNER.append(0)
                        date = {'year':normalizedNER[0], 'month':normalizedNER[1], 'day':normalizedNER[2]}
                    except:
                        date = {'year':0, 'month':0, 'day':0}
        return myDate(date, original_text = text, version = 0)

#money
    if entity['ner'] == 'MONEY':
        value = getNumberFromString(normalizedNER)[0]
        return money(value = value, original_text = text, version = 0)
    
#time
    if entity['ner'] == 'TIME':
        if normalizedNER=="":
            value = [ 0, 0, 0]
        else :
            text = text.lower()
            if 'night' in text or 'afternoon' in text or 'morning' in text or 'evening' in text :
                value = [0,0,0]
            dateAndTime = normalizedNER.split('T')
            if len(dateAndTime)<2:
                value = [0,0,0]
            try:
                temp = dateAndTime[1]
                temp = temp.split(':')
                temp[1] = temp[1].split('+')[0]
                temp[1] = temp[1].split('-')[0]
                value = [int(temp[0]),int(temp[1]),0]
            except:
                value = [ 0, 0, 0]
        return time(value, original_text=text, version=0)
    
    raise Exception(f"can't parse {entity}")


class number(object):
    def __init__(self, value, original_text="", version = 1):
        self.value = value
        self.original_text = original_text
        self.version = version
    
    def __str__(self):
        if self.version == 0 and False :
            return self.original_text
        #
        if abs(round(self.value)-self.value) <eps:
            return str(round(self.value))
        #
        return str(round(self.value, 2))

    #
    def __add__(self, other):# +
        if isinstance(other, number):
            return number(self.value + other.value)
        if isinstance(other, percentage):
            return percentage(self.value * 100 + other.value,)
        if isinstance(other, money):
            return money(self.value+ other.value, )
        if isinstance(other, ordinal):
            return number(self.value+other.value,)

        if isinstance(other, time):
            raise Exception("can't support __add__ with number:{} time:{}".format(self, other))
        if isinstance(other, myDate):
            return other + self
        if isinstance(other, deltaDate): 
            return other + self
        if isinstance(other, deltaTime):
            raise Exception("can't support __add__ with number:{} time:{}".format(self, other))
            #return other + self
        raise TypeError("can't __add__ with number:{} and unsupport type:{}:{}".format(self,type(other), other))
    
    def __sub__(self, other):# -
        if isinstance(other, number):
            return number(abs(self.value - other.value),version = 1)
        if isinstance(other, percentage):
            return percentage(abs(self.value * 100 - other.value),version = 1)
        if isinstance(other, money):
            return money(abs(self.value- other.value), version = 1)
        if isinstance(other, ordinal):
            return number(abs(self.value-other.value), version = 1)

        if isinstance(other, time):
            return other - self
            #raise Exception("can't support __sub__ with number:{} time:{}".format(self, other))
        if isinstance(other, myDate):
            return other - self
            #raise Exception("can't support __sub__ with number:{} myDate:{}".format(self, other))
        if isinstance(other, deltaDate):
            return other - self
        if isinstance(other, deltaTime):
            return other - self
        raise TypeError("can't __sub__ with number:{} and unsupport type:{}:{}".format(self,type(other), other))
    
    def __eq__(self, other):# = 
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return abs(self.value - other.value)<eps
        if isinstance(other, percentage):
            return abs(self.value*100.0 - other.value)<eps
        

        if isinstance(other, deltaDate):
            kk = other.getUsefulKey()
            return abs(self.value- other.date[kk]) < eps
        if isinstance(other, deltaTime):
            kk = other.getUsefulKey()
            return abs(self.value- other.time[kk]) < eps
        
        if isinstance(other, myDate):
            kk = other.getUsefulKey()
            return abs(self.value - other.date[kk]) < eps
        if isinstance(other, time):
            kk = other.getUsefulKey()
            return abs(self.value- other.time[kk]) < eps

        raise TypeError("can't __eq__ with number:{} and unsupport type:{}:{}".format(self,type(other),other))
    
    def __lt__(self, other):# <
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return self.value < other.value
        if isinstance(other, percentage):
            return self.value < other.value/100
        if isinstance(other, myDate) or isinstance(other, deltaDate):
            kk = other.getUsefulKey()
            return self.value < other.date[kk]
        if isinstance(other, time) or isinstance(other, deltaTime):
            kk = other.getUsefulKey()
            return self.value < other.time[kk]

        raise TypeError("can't __lt__ with number:{} and unsupport type:{}:{}".format(self,type(other),other))
    
    def __truediv__(self, other):
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            if abs(other.value) < eps:
                return number(0.0)
            return number(self.value/other.value)
        if isinstance(other, percentage):
            return number(self.value/other.value*100)
        
        
        raise TypeError("can't __truediv__ with number:{} and unsupport type:{}:{}".format(self,type(other),other))
    
    def __mul__(self, other):
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return number(self.value * other.value)
        if isinstance(other, percentage):
            return number(self.value * other.value / 100)
        raise TypeError("can't __truediv__ with number:{} and unsupport type:{}:{}".format(self,type(other),other))

class percentage(object):
    def __init__(self, value, original_text="", version=1):
        self.value = value
        self.original_text = original_text
        self.version = version
    def __str__(self):
        if self.version == 0 and False :
            return self.original_text
        if abs(round(self.value)-self.value) <eps:
            return str(round(self.value))
        return str(round(self.value, 2))

    def __add__(self, other):# +
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return percentage(self.value + other.value*100)
        if isinstance(other, percentage):
            return percentage(self.value + other.value)
        raise TypeError("can't __add__ with percentage:{} and unsupport type:{}:{}".format(self,type(other), other))

    def __sub__(self, other):# -
        if isinstance(other, number)  or isinstance(other, ordinal):
            return percentage(abs(self.value - other.value*100))
        if isinstance(other, percentage):
            return percentage(abs(self.value - other.value))
        raise TypeError("can't __sub__ with percentage:{} and unsupport type:{}:{}".format(self,type(other), other))

    def __eq__(self, other):# = 
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return abs(self.value - other.value * 100) < eps
        if isinstance(other,percentage):
            return abs(self.value - other.value) < eps
        
        raise TypeError("can't __eq__ with percentage:{} and unsupport type:{}:{}".format(self,type(other), other))
        
    def __lt__(self, other):# <
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return self.value < (other.value * 100)
        if isinstance(other, percentage):
            return self.value < other.value
        raise TypeError("can't __lt__ with percentage:{} and unsupport type:{}:{}".format(self,type(other), other))
    
    def __mul__(self, other):
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return number(self.value/100*other.value)
        raise TypeError("can't __mul__ with percentage:{} and unsupport type:{}:{}".format(self,type(other), other))

class money(object):
    def __init__(self, value, original_text = "", version = 1):
        self.value = value
        self.original_text = original_text
        self.version = version
    
    def __str__(self):
        if self.version == 0 and False :
            return self.original_text
        if abs(round(self.value)-self.value) <eps:
            return str(round(self.value))
        return str(round(self.value, 2))

    def __add__(self, other):# +
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return money(self.value + other.value)
        raise TypeError("can't __add__ with money:{} and unsupport type:{}:{}".format(self,type(other), other))

    def __sub__(self, other):# -
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return number(abs(self.value - other.value))
        raise TypeError("can't __sub__ with money:{} and unsupport type:{}:{}".format(self,type(other), other))
    
    def __eq__(self, other):# = 
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return abs(self.value - other.value)<eps
        raise TypeError("can't __eq__ with money:{} and unsupport type:{}:{}".format(self,type(other), other))
    
    def __lt__(self, other):# <
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return self.value < other.value
        raise TypeError("can't __lt__ with money:{} and unsupport type:{}:{}".format(self,type(other), other))
    
    def __mul__(self, other):# *
        if isinstance(other, number) or isinstance(other, money) or isinstance(other, ordinal):
            return number(self.value * other.value)
        if isinstance(other, percentage):
            return money(self.value * other.value / 100)
        raise TypeError("can't __mul__ with money:{} and unsupport type:{}:{}".format(self,type(other), other))

class ordinal(object):
    def __init__(self, value, original_text="",version = 1 ):
        self.value = value
        self.original_text = original_text
        self.version = version

    def __str__(self):
        if self.version == 0 and False :
            return self.original_text
        if abs(round(self.value)-self.value) <eps:
            return str(round(self.value))
        return str(round(self.value, 2))

    
    def __add__(self, other):# +
        if isinstance(other, number) or isinstance(other,ordinal):
            return number(self.value + other.value)            
        raise TypeError("can't __add__ with ordinal:{} and unsupport type:{}:{}".format(self,type(other), other))

    def __sub__(self, other):# -
        if isinstance(other, number) or isinstance(other,ordinal):
            return number(abs(self.value - other.value))
        raise TypeError("can't __sub__ with ordinal:{} and unsupport type:{}:{}".format(self,type(other), other))
    
    def __eq__(self, other):# = 
        if isinstance(other, number) or isinstance(other,ordinal):
            return abs(self.value - other.value) < eps
        raise TypeError("can't __eq__ with ordinal:{} and unsupport type:{}:{}".format(self,type(other), other))
    
    def __lt__(self, other):# <
        if isinstance(other, number) or isinstance(other,ordinal):
            return self.value < other.value
        raise TypeError("can't __lt__ with ordinal:{} and unsupport type:{}:{}".format(self,type(other), other))
    
    def __mul__(self, other):# *
        if isinstance(other, number):
            return number(self.value * other.value)
        raise TypeError("can't __mul__ with ordinal:{} and unsupport type:{}:{}".format(self,type(other), other))



class deltaDate(object): 
    def __init__(self, value, original_text = "",version = 1): #若其存在多值，则仅用于输出。单值仅用于计算。
        if isinstance(value, list):
            year, month, day = value[0],value[1],value[2]
        elif isinstance(value, dict):
            year, month, day = value['year'],value['month'],value['day']
        else:
            year, month, day = 0, 0, 0
        self.date = {'year' : year, 'month' : month, 'day' : day}
        self.original_text = original_text
        self.version = version

        self.month_delta = -1
        self.year_delta = -1
        #
    def getUsefulKey(self):
        usefulKey = [k for k,v in self.date.items() if abs(v)>eps]
        if len(usefulKey) >= 1:
            return usefulKey[0]
        else:
            return 'year'

    def __str__(self, pattern = ""):
        if self.version == 0 and False :
            return self.original_text

        pattern = re.sub('es','',pattern)
        pattern = re.sub('s','',pattern)

        if pattern == "month":
            return str(round(self.date['year']*12 + self.date['month']))
        elif pattern == 'year' or pattern == 'day':
            return str(round(self.date[pattern]))
        else:
            #if self.year_delta != -1:
            #    return self.year_delta
            
            add_month = self.date['day'] // 30
            self.date['day'] = self.date['day'] % 30
            self.date['month'] += add_month

            add_year = self.date['month'] // 12
            self.date['month'] = self.date['month'] % 12
            self.date['year'] += add_year

            if self.date['day']!=0 and self.date['year']==0 and self.date['month']==0:
                if self.date['day'] > 1:
                    return str(round(self.date['day'])) + " " + 'days'
                else :
                    return str(round(self.date['day'])) + " " + 'day'
            
            if self.date['year']!=0 and self.date['month']==0 and self.date['day']==0:
                return str(round(self.date['year']))
            if self.date['year']==0 and self.date['month']!=0 and self.date['day']==0:
                return str(round(self.date['month']))
            
            for k,v in self.date.items():
                if round(v)!=0:
                    return str(round(v))





            return '0'
    def to_month(self):
        if self.month_delta != -1:
            return round(self.month_delta)
        num = self.date['month']
        num += self.date['year']*12 + self.date['day']/30.0
        return round(num)
    def to_year(self):
        if self.year_delta != -1:
            return round(self.year_delta)
        num = self.date['year']
        num += math.ceil(self.date['year']/12.0+self.date['day']/365.0)
        return round(num)
    def __add__(self, date):
        if isinstance(date,myDate): #
            usefulKey = [key for key,value in self.date.items() if abs(value)>0]
            if len(usefulKey) > 1:#
                raise Exception("Unsupported add with deltaDate and date!")
            if len(usefulKey) == 0:
                return date
            usefulKey = usefulKey[0]
            if usefulKey=='day' and date.is_complete(): #
                if date.date['year'] != 0:
                    default_date = datetime.date(date.date['year'], date.date['month'], date.date['day'])
                    add_date = datetime.timedelta(days=self.date['day'])
                    result_date = default_date + add_date
                    return myDate([result_date.year,result_date.month,result_date.day])
                else :#
                    default_date = datetime.date(2020, date.date['month'], date.date['day'])
                    add_date = datetime.timedelta(days=self.date['day'])
                    result_date = default_date + add_date
                    return myDate([0,result_date.month,result_date.day]) 
            else:#
                newDate = myDate([date.date['year'],date.date['month'],date.date['day']])
                newDate.date['month'] += self.date['month']
                newDate.date['year'] += self.date['year']
                newDate.date['day'] += self.date['day']
                while newDate.date['day'] > 30:
                    newDate.date['day'] -= 30
                    newDate.date['month'] += 1
                while newDate.date['month'] > 12:
                    newDate.date['month'] -= 12
                    newDate.date['year'] += 1
                for key,value  in  newDate.date.items():
                    if key not in usefulKey:
                        newDate.date[key] = 0
                return newDate
        elif isinstance(date, deltaDate):
            myUsefulKey = [key for key,value in self.date.items() if value != 0]
            otherUsefulKey = [key for key,value in date.date.items() if value != 0]
            newDate = deltaDate(self.date['year'],self.date['month'],self.date['day'])
            for key in ['year','month','day']:
                if key in myUsefulKey and key in otherUsefulKey:
                    newDate.date[key] += date.date[key]
                else :
                    newDate.date[key] = 0
            return newDate
        elif isinstance(date, number):
            key = self.getUsefulKey()
            return number(abs(self.date[key] + date.value))
        else :
            raise Exception(f"Unsupported add with {type(self)} and {type(date)}")
    
    def __sub__(self, date):
        if isinstance(date,myDate): #
            usefulKey = [key for key,value in self.date.items() if value != 0]
            if len(usefulKey) > 1:
                raise Exception("Unsupported add with deltaDate and date!")
            if len(usefulKey) == 0:
                return date
            usefulKey = usefulKey[0]
            if usefulKey == 'day' and date.is_complete():
                if date.date['year'] != 0:
                    default_date = datetime.date(date.date['year'], date.date['month'], date.date['day'])
                    add_date = datetime.timedelta(days=self.date['day'])
                    result_date = default_date - add_date
                    return myDate([result_date.year,result_date.month,result_date.day])
                else :
                    default_date = datetime.date(2020, date.date['month'], date.date['day'])
                    add_date = datetime.timedelta(days=self.date['day'])
                    result_date = default_date - add_date
                    return myDate([0,result_date.month,result_date.day])
            else:
                newDate = myDate([date.date['year'],date.date['month'],date.date['day']])
                usefulKey = [key for key,value in newDate.date.items() if value != 0]
                newDate.date['month'] -= self.date['month']
                newDate.date['year'] -= self.date['year']
                newDate.date['day'] -= self.date['day']
                while newDate.date['day'] < 1:
                    newDate.date['day'] += 30
                    newDate.date['month'] -= 1
                while newDate.date['month'] < 1 :
                    newDate.date['month'] += 12
                    newDate.date['year'] -= 1
                for key,value  in  newDate.date.items():
                    if key not in usefulKey:
                        newDate.date[key] = 0
                return newDate
        elif isinstance(date, deltaDate):
            #logger.info("FLAG! {} sub {}".format(self,date))
            myUsefulKey = [key for key,value in self.date.items() if value != 0]
            otherUsefulKey = [key for key,value in date.date.items() if value != 0]
            self_date = deltaDate([self.date['year'],self.date['month'],self.date['day']])
            other_date = deltaDate([date.date['year'],date.date['month'],date.date['day']])
            if self_date < other_date :
                self_date,other_date = other_date,self_date
            newDate = deltaDate([self_date.date['year'],self_date.date['month'],self_date.date['day']])
            for key in ['year','month','day']:
                if key in myUsefulKey and key in otherUsefulKey:
                    newDate.date[key] -= other_date.date[key]
                else :
                    newDate.date[key] = 0
            
            while newDate.date['day'] < 0:
                newDate.date['day'] += 30
                newDate.date['month'] -= 1
            while newDate.date['month'] < 0:
                newDate.date['month'] += 12
                newDate.date['year'] -= 1
            return newDate
        elif isinstance(date, number):
            key = self.getUsefulKey()
            return number(abs(self.date[key]-date.value))
        else :
            raise Exception(f"Unsupported add with {type(self)} and {type(date)}")

    def __lt__(self, other):
        if isinstance(other, number):
            if self.date['year'] != 0 :
                return self.date['year']< other.value
            if self.date['month'] != 0 :
                return self.date['month']< other.value
            if self.date['day'] != 0 :
                return self.date['day']< other.value
            return True
        if isinstance(other, deltaDate):
            if self.date['year'] != other.date['year']:
                return self.date['year'] < other.date['year']
            if self.date['month'] != other.date['month']:
                return self.date['month'] < other.date['month']
            if self.date['day'] != other.date['day']:
                return self.date['day'] < other.date['day']
            return True
        raise TypeError("Cannot __lt__ with {} and {}".format(type(self), type(other)))

def valid_date(num):
    return abs(num)>eps
class myDate(object):
    def __init__(self, value, original_text="", version=1):
        if isinstance(value, list):
            year, month, day = value[0],value[1],value[2]
        elif isinstance(value, dict):
            year, month, day = value['year'],value['month'],value['day']
        else:
            year, month, day = 0, 0, 0
        self.date = {'year' : year, 'month' : month, 'day' : day}
        self.original_text = original_text
        self.version = version
    def getUsefulKey(self):
        usefulKey = [k for k,v in self.date.items() if abs(v)>eps]
        if len(usefulKey) >= 1:
            return usefulKey[0]
        else:
            return 'year'
    def is_complete(self):
        if self.date['month']> eps and self.date['day']> eps:
            return True
        else:
            return False
    def __str__(self):
        dateList = [self.date['day'], self.date['month'], self.date['year']]
        string = ""
        for idx, num in enumerate(dateList):
            if num != 0:
                add_str = str(num)
                if idx == 1:
                    add_str = month_table[num - 1]
                string += add_str+' '

        if string == "":
            string = "No Date"
        else :
            string = string[:-1]
        return string
    
    def __add__(self, other):
        if isinstance(other, myDate):
            try:
                delta = deltaDate(**other.date)
                result = delta + self
                return result
            except:
                pass
            try:
                delta = deltaDate(**self.date)
                result = delta + other
                return result
            except:
                pass
            return self
        if isinstance(other,deltaDate):
            return other + self
        if isinstance(other, number):
            kk = self.getUsefulKey()
            return number(self.date[kk]+other.value)
        raise TypeError("can't __add__ {} and {}".format(self, other))
    
    
    def __sub__(self, other):
        if isinstance(other, myDate):
            if self.is_complete() and other.is_complete():
                if not valid_date(self.date['year']) or not valid_date(other.date['year']):#
                    self_date = datetime.date(2020,self.date['month'],self.date['day'])
                    other_date = datetime.date(2020,other.date['month'],other.date['day'])
                    delta_year = 0
                else :
                    self_date = datetime.date(self.date['year'] ,self.date['month'],self.date['day'])
                    other_date = datetime.date(other.date['year'] ,other.date['month'],other.date['day'])
                    delta_year = abs(self.date['year'] - other.date['year'])
                if self_date < other_date:
                    other_date, self_date = self_date, other_date
                delta_date = self_date - other_date
                delta_date = deltaDate({'year': 0, 'month': 0, 'day': delta_date.days})
                delta_date.month_delta = (self_date.year - other_date.year) * 12 + self_date.month - other_date.month
                delta_date.year_delta = delta_year
                return delta_date
            else :
                self_date = deltaDate({'year' : self.date['year'] ,'month' : self.date['month'],'day' :  self.date['day'] } )
                other_date = deltaDate({'year' : other.date['year'] ,'month' : other.date['month'],'day' :  other.date['day'] } )
                
                if not valid_date(self_date.date['year']) or not valid_date(other_date.date['year']):
                    self_date.date['year'],other_date.date['year'] = 0,0
                if not valid_date(self_date.date['month']) or not valid_date(other_date.date['month']):
                    self_date.date['month'],other_date.date['month'] = 0,0
                if not valid_date(self_date.date['day']) or not valid_date(other_date.date['day']):
                    self_date.date['day'],other_date.date['day'] = 0,0
                
                if other_date < self_date:
                    delta_date = deltaDate({'year' : self_date.date['year'] - other_date.date['year'],'month' : self_date.date['month'] - other_date.date['month'],'day' :  self_date.date['day'] - other_date.date['day'] } )
                else :
                    delta_date = deltaDate({'year' : other_date.date['year'] - self_date.date['year'],'month' : other_date.date['month'] - self_date.date['month'],'day' :  other_date.date['day'] - self_date.date['day'] } )
                

                while delta_date.date['day']< 0:
                    delta_date.date['day'] += 30
                    delta_date.date['month'] -= 1
                
                while delta_date.date['month']< 0:
                    delta_date.date['month'] += 12
                    delta_date.date['year'] -= 1
                if abs(self.date['year']) > eps and abs(other.date['year']) > eps:
                    delta_date.year_delta = abs(self_date.date['year'] - other_date.date['year'])
                if abs(self.date['month']) > eps and abs(other.date['month']) > eps:
                    delta_date.month_delta = abs((self_date.date['year'] - other_date.date['year'])*12 + self_date.date['month'] - other_date.date['month'])
                #logger.info("delta_date:{}".format(delta_date))
                return delta_date
        if isinstance(other, deltaDate):
            return other - self
        if isinstance(other, number):
            for key ,value in self.date.items():
                if value != 0:
                    newDate = myDate([0,0,0])
                    newDate.date[key] = round(other.value)
                    return self - newDate
        

        raise TypeError("can't __sub__ {} and {}".format(self, other))
    def __lt__(self, other):
        if isinstance(other, myDate):
            if self.date['year']!=other.date['year']:
                return self.date['year']<other.date['year']
            if self.date['month']!=other.date['month']:
                return self.date['month']<other.date['month']
            if self.date['day']!=other.date['day']:
                return self.date['day']<other.date['day']
            return True
        if isinstance(other, number):
            if self.date['year']!=0:
                return self.date['year']<other.value
            if self.date['month']!=0:
                return self.date['month']<other.value
            if self.date['day']!=0:
                return self.date['day']<other.value
            return True
        raise TypeError("can't __lt__ {} and {}".format(self, other))


class deltaTime(object):
    def __init__(self, value, original_text = "", version = 1):
        if isinstance(value, list):
            hour, minute, second = value[0],value[1],value[2]
        elif isinstance(value, dict):
            hour, minute, second = value['hour'],value['minute'],value['second']
        else:
            hour, minute, second = 0, 0, 0
        self.time = {'hour' : hour, 'minute' : minute, 'second' : second}        
        self.original_text = original_text
        self.version = version
    def getUsefulKey(self):
        usefulKey = [k for k,v in self.time.items() if abs(v)<= eps]
        if len(usefulKey) >= 1:
            return usefulKey[0]
        else:
            return 'hour'
    def __str__(self):
        return str(self.time['hour']) + ':' + str(self.time['minute']) + ':' + str(self.time['second'])


class time(object):# self.time [hour, minute, second]
    def __init__(self, value, original_text = "", version = 1):
        if isinstance(value, list):
            hour, minute, second = value[0],value[1],value[2]
        elif isinstance(value, dict):
            hour, minute, second = value['hour'],value['minute'],value['second']
        else:
            hour, minute, second = 0, 0, 0
        self.time = {'hour' : hour, 'minute' : minute, 'second' : second}        
        self.original_text = original_text
        self.version = version
    
    def getUsefulKey(self):
        usefulKey = [k for k,v in self.time.items() if abs(v)<= eps]
        if len(usefulKey) >= 1:
            return usefulKey[0]
        else:
            return 'hour'

    def __str__(self):
        for num in self.time:
            if num != 0:
                return str(num)
        return "0"    
    
    def __eq__(self, other):
        if isinstance(other, time):
            return other.time['hour'] == self.time['hour'] and other.time['minute'] ==self.time['minute'] and other.time['second'] ==self.time['second']
        raise TypeError("can't __eq__ with {} and {}".format(self, other))
    
    def __lt__(self, other):
        if isinstance(other, time):
            if other.time['hour'] != self.time['hour']:
                return self.time['hour'] < other.time['hour']
            if other.time['minute'] != self.time['minute']:
                return self.time['minute'] < other.time['minute']
            return self.time['second'] < other.time['second']
        raise TypeError("can't __lt__ with {} and {}".format(self, other))

    def __sub__(self, other):
        if isinstance(other,time):
            temp1 = datetime.datetime(2020,1,1,self.time['hour'],self.time['hour'],self.time['second'])
            temp2 = datetime.datetime(2020,1,1,other.time['hour'],other.time['hour'],other.time['second'])
            if temp1<temp2:
                temp1, temp2 = temp2, temp1
            delta = temp1 - temp2
            newTime = [0, 0, delta.seconds] 
            return deltaTime(newTime)
        raise Exception("Can't support __sub__ of time and {}".format(type(other)))







if __name__=='__main__':
    pass

