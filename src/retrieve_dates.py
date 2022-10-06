

import re



MONTHS={v.lower():i+1 for i,v in enumerate(["January","February", "March","April","May","June","July","August","September","October","November","December"])}
months = ["","January","February", "March","April","May","June","July","August","September","October","November","December"]
FLAG_NER = "@"
FLAG_SENTENCE = "##"
NUM_NER_TYPES = ['ENT2NUM', 'NUMBER', 'PERCENT','MONEY','TIME','DATE','DURATION','ORDINAL', 'YARD']
FLAG_DATE = '@DATE@'
FLAG_NUMBER = '@NUMBER@'
LOWER_FLAG_DATE = '@date@'
NUMBER_NER_TYPE = "NUMBER"
YARD_NER_TYPE = "YARD"

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def remove_ner_type(token):
  pos = token.find('@')
  if pos > 0:
    return token[:pos]
  return token


def get_day(token):
  token = remove_ner_type(token)
  m = re.match("^(\d{1,2})(th|st|nd|rd)?$", token)
  if m:
    if int(m.group(1)) < 32:
      return m.group(1)
  return None

def get_year(token):
  token = remove_ner_type(token)
  if re.match("^\d{4}$", token) and int(token) <= 2020:
    return token
  return None

def get_month(token):
  token = remove_ner_type(token)
  if token in MONTHS.keys():
    return MONTHS[token]
  return None


def normalize_day_month(content):
  return str(content) if len(content) >=2 else '0'+str(content)

def retrieve_dates(passage_text):
  prev_is_whitespace, raw_tokens,word_to_char_offset = True, [], []
  for i, c in enumerate(passage_text):
      if is_whitespace(c):  # or c in ["-", "–", "~"]:
          prev_is_whitespace = True
      elif c in ["-", "–", "~"]:
          raw_tokens.append(c)
          word_to_char_offset.append(i)
          prev_is_whitespace = True
      else:
          if prev_is_whitespace:
              raw_tokens.append(c)
              word_to_char_offset.append(i)
          else:
              raw_tokens[-1] += c
          prev_is_whitespace = False
  tokens = [token.lower() for token in raw_tokens]

  cyear = None#allow using from previous sentences
  def search(i, tokens, is_before, is_day):
      j = i
      while True:
       if (is_before and j < 0) or  (not is_before and j >= len(tokens)):
         break
       if remove_ner_type(tokens[j]) in ['and', 'or', "-", "–", "~", ',', 'of','by']:
         if is_before:
           j -= 1
         else:
           j += 1
         continue
       else:
         if (is_day and get_day(tokens[j])) or (not is_day and get_year(tokens[j]) is not None):
           return j
         break 
      return None

  dates = []
  for i,token in enumerate(tokens):
      month = get_month(token)
      #if month!=None:
      #  month = months[month]
      if month is not None:
        sidx,eidx = i,i#closure[]
        sidx2,eidx2 = None, None#closure[]
        day,year, day2=None,None,None
        idx = search(i-1, tokens, is_before=True, is_day=True)
        if idx and (len(dates)<=0 or idx > dates[-1][1]):
          sidx=idx
          day = get_day(tokens[sidx])
          idx = search(i+1, tokens, is_before=False, is_day=False)
          if idx:
            year=get_year(tokens[idx])
            eidx=idx
          sidx2 = search(i-2, tokens, is_before=True, is_day=True)

        else:
          idx = search(i+1, tokens, is_before=False, is_day=True)
          if idx:
            day = get_day(tokens[idx])
          if day:
            eidx=idx
            idx = search(i+2, tokens, is_before=False, is_day=False)
            if idx:
              year=get_year(tokens[idx])
              eidx=idx
          else:
            idx = search(i+1, tokens, is_before=False, is_day=False)
            if idx:
              year=get_year(tokens[idx])
              eidx=idx

        if year is None and day is None and i>0 and tokens[i-1] not in ['in', 'by', 'on', 'of'] and not (tokens[i-1]=='between' and tokens[i+1]=='and'):
          continue
        if year is None and cyear is not None:#miss backward
          #print('use replace',(sidx, eidx, year, month, day))
          year = cyear
        if day is None:
          day = 1
          #print('use default day')
        dates.append((sidx, eidx, year, month, day,' '.join(tokens[sidx:eidx+1])))
        if sidx2 is not None:# and sidx2 > dates[-1][1]:
          #print('match!!!!!!', ' '.join(tokens[sidx2:eidx+1]))
          dates.append((sidx2, sidx2, year, month, int(get_day(tokens[sidx2]))))
      else:
        cyear = get_year(tokens[i])
        if cyear is not None:
          cyear=int(cyear)
          for j,date in enumerate(dates):
            if date[2] is None:
              #print('use latter year',date)
              dates[j]=(date[0],date[1],cyear, date[3],date[4])

  default_year=2020# run nian
  for j,date in enumerate(dates):
    if date[2] is None:
       dates[j]=(date[0],date[1],default_year, date[3],date[4])

  if len(dates) <= 1:
    return passage_text

  res_tokens, didx, date_indices, date_tokens = [], 0, [], []
  dates = sorted(dates, key=lambda x: x[0])
  for i in range(len(tokens)):
    token = tokens[i]
    raw_token = raw_tokens[i]
    if didx < len(dates):
      date = dates[didx]
      if i >= date[0]:
        if i <= date[1]:
          if i==date[1]:
            date_indices.append(len(res_tokens))
            date_tokens.append(str(date[2])+normalize_day_month(str(date[3]))+normalize_day_month(str(date[4])))
#            res_tokens.append(date_tokens[-1]+FLAG_DATE)
            res_tokens.append(str(date[2]) )#+FLAG_DATE)
            res_tokens.append(normalize_day_month(str(date[3])))#+FLAG_DATE)
            res_tokens.append(normalize_day_month(str(date[4])))#+FLAG_DATE)
            didx+=1
          continue
    res_tokens.append(raw_token)

  return ' '.join(res_tokens)
 
