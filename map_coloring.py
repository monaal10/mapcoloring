from matplotlib import pyplot
import cv2
import numpy
import sys

colors = [(87, 253, 214), (115, 213, 111), (153, 51, 255), (240, 110, 50)]
rowDirections = list()
rowDirections.append(-1)
rowDirections.append(+1)


def validateIndex(i, j):
    if i < 0 or i >= w or j < 0 or j >= h:
        return f
    return t


bgCLR = [255] * 3
bgCLR = tuple(bgCLR)
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
h = len(img)
w = len(img[0])
area = w * h
symbol = list()
for m in range(h):
    symbol.append([])
    for n in range(w):
        symbol[m].append(-1)
states = list()

districts = list()
for k in range(1080):
    districts.append([])

areaBorder = list()
for k in range(1080):
    areaBorder.append([])

colorLog = list()
for k in range(1000):
    colorLog.append(-1)

columnDirections = list()

for k in range(2):
    rowDirections.append(0)
    columnDirections.append(0)


columnDirections.append(-1)
columnDirections.append(+1)


class Region:
    def __init__(this, region_id, region_i, region_j):
        this.id = region_id
        this.i = region_i
        this.j = region_j
        this.neighbor = []

    def connect(this, region):
        this.neighbor.append(region.id)


def neighbors(region1: Region, region2: Region):
    tm1, tm2 = region1.i, region1.j
    nm1, nm2 = region2.i, region2.j
    misq = float('inf')
    for i in areaBorder[symbol[tm2][tm1]]:
        for j in areaBorder[symbol[nm2][nm1]]:
            temp = (i[0] - j[0]) * (i[0] - j[0]) + \
                (i[1] - j[1]) * (i[1] - j[1])
            if temp < misq:
                misq = temp
                tm1, tm2 = i[0], i[1]
                nm1, nm2 = j[0], j[1]
    dr, dc = nm1 - tm1, nm2 - tm2
    if abs(dr) + abs(dc) <= 1:
        return t
    dr, dc = float(dr), float(dc)

    if misq >= 0.15 * (arr):
        return False
    limit = int(2 * ((arr) ** (1/2)))
    k = 0
    div1 = dr / limit
    div2 = dc / limit
    while k < limit:
        i = int(tm1 + k * div1 + 1 / 2)
        j = int(tm2 + k * div2 + 1 / 2)
        if symbol[j][i] >= 0 and (i != tm1 or j != tm2) and (i != nm1 or j != nm2):
            return f
        k = k + 1
    return t


def connectRegions():
    for s1 in range(len(states)):
        for s2 in range(len(states)):
            if s2 > s1 and neighbors(states[s1], states[s2]):
                states[s1].connect(states[s2])
                states[s2].connect(states[s1])


def amgMC(region):
    if region == len(states):
        for i in range(len(states)):
            colorizeRegion(states[i], colors[colorLog[i]])
        z, y, x = cv2.split(img)
        frame_rgb = cv2.merge((x, y, z))
        pyplot.imshow(frame_rgb)
        pyplot.title("Output")
        pyplot.show()
    for i in range(len(colors)):
        flag = t
        for j in states[region].neighbor:
            if colorLog[j] == i:
                flag = f
                break
        if flag:
            colorLog[region] = i
            amgMC(region + 1)
            colorLog[region] = -1


def distPxls():
    for i in range(h):
        for j in range(w):
            region_symbol = symbol[i][j]
            districts[region_symbol].append((j, i))
            if borderCheck(j, i):
                areaBorder[region_symbol].append((j, i))


t = True
f = False


def graphStates():
    for i in range(h):
        for j in range(w):
            if symbol[i][j] == -1:
                ca = regSz(j, i, -1, len(states))
                if ca > 0.0006 * area:
                    states.append(Region(len(states), j, i))
                else:
                    regSz(j, i, len(states), -1)
    distPxls()


def borderCheck(i, j):
    if symbol[j][i] == -2:
        return False

    k = 0
    while k < 4:
        b1 = i + rowDirections[k]
        b2 = j + columnDirections[k]
        if validateIndex(b1, b2) and symbol[b2][b1] == -2:
            return t
        k = k + 1
    return f


def colorMatcher(i1, j1, i2, j2):
    if not validateIndex(i1, j1) or not validateIndex(i2, j2):
        return f
    z1, y1, x1 = img[j1][i1]
    z2, y2, x2 = img[j2][i2]
    x1, y1, z1 = int(x1), int(y1), int(z1)
    x2, y2, z2 = int(x2), int(y2), int(z2)
    diff = abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)
    return diff <= 150


def findBackground():
    for i in range(h):
        for j in range(w):
            x, y, z = img[i][j]
            x, y, z = int(x), int(y), int(z)
            s = x + y + z
            if s < 100:
                img[i][j] = bgCLR
                symbol[i][j] = -2
            if s > 663:
                img[i][j] = bgCLR
                symbol[i][j] = -2


def regSz(tm, nm, sym1, sym2):
    if not validateIndex(tm, nm) or symbol[nm][tm] != sym1:
        return 0
    regArea = 0
    q = [(tm, nm)]
    symbol[nm][tm] = sym2
    while q:
        i, j = q.pop(0)
        symbol[j][i] = sym2
        regArea = regArea + 1
        k = 0
        while k < 4:
            i2 = i + rowDirections[k]
            j2 = j + columnDirections[k]
            if validateIndex(i2, j2) and symbol[j2][i2] == sym1 and colorMatcher(i, j, i2, j2):
                symbol[j2][i2] = sym2
                q.append((i2, j2))
            k = k + 1
    return regArea


arr = w * w + h * h


def colorizeRegion(region: Region, clr):
    region_idx = symbol[region.j][region.i]
    for i in range(len(districts[region_idx])):
        p1 = districts[region_idx][i][0]
        p2 = districts[region_idx][i][1]
        img[p2][p1] = clr


def changeBG():
    for y in range(h):
        for x in range(w):
            if symbol[y][x] == -1 or symbol[y][x] == -2:
                img[y][x] = bgCLR


if __name__ == '__main__':
    findBackground()
    img = cv2.medianBlur(img, 3)
    findBackground()
    img = cv2.filter2D(
        img, -1, numpy.array([[-1]*3, [-1, 9, -1], [-1]*3]))
    findBackground()
    graphStates()
    connectRegions()
    changeBG()
    try:
        amgMC(0)
    except Exception:
        print("Map Printed")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
