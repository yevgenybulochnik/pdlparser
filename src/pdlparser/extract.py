import pytesseract


def extract_text(img_array, rect):
    x, y, w, h = rect
    crop = img_array[y:y+h, x:x+w]

    text = pytesseract.image_to_string(crop, )

    clean = []

    for text in text.split('\n'):
        if text == '\x0c':
            continue
        if text:
            clean.append(text.strip())

    return clean


def extract_set(img_array, class_group):
    header = class_group['header']
    preferred = class_group['left'][1]
    non_preferred = class_group['right'][1]

    class_set = {}

    class_set['header'] = extract_text(img_array, header)
    class_set['preferred'] = extract_text(img_array, preferred)
    class_set['non-preferred'] = extract_text(img_array, non_preferred)

    return class_set
