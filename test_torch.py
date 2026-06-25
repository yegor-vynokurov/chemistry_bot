import torch
from typing import List
from itertools import chain
from typing import Optional


def main() -> None:
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("GPU is not available to PyTorch.")
        return

    device = torch.device("cuda")
    print("GPU device:", torch.cuda.get_device_name(0))

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = x * 2

    print("Tensor x:", x)
    print("Tensor y:", y)
    print("x device:", x.device)
    print("y device:", y.device)

# MAYBE NEED FOR THE PATTERNS COMPARING IN THE PRICES MOVING: 
# WE MAY COMPARE LENGHT OF THE UP AND DOWN PRICES LENGHTS
class Solution:
    def findLengthOfLCIS_0(self, nums: List[int]) -> int:
        ans = 1
        temp = 1

        for i in range(1,len(nums)):
            print('nums[i] > nums[i-1]', nums[i] > nums[i-1])
            if nums[i] > nums[i-1]:
                temp += 1
            else:
                temp = 1
            ans = max(ans, temp)

        return ans

class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        ans = 0
        temp = 0
        prev = -float('inf')

        for num in nums:
            if num > prev:
                temp += 1
                ans = max(ans, temp)
            else:
                temp = 1
            prev = num

        return ans
    

# it is may be good for timeseries relative prices for finding points that break the trend:
class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        temp = set()
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                temp.add(0)
            elif nums[i] >= nums[i-1]:
                temp.add(1)
            elif nums[i] <= nums[i-1]:
                temp.add(-1)
            if -1 in temp and 1 in temp:
                return False

        return True
    

class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:

        def inc(nums):
            prev = -float('inf')
            for num in nums:
                if num < prev:
                    return False
                prev = num

            return True

        def dec(nums):
            prev = float('inf')
            for num in nums:
                if num > prev:
                    return False
                prev = num

            return True
            
        return inc(nums) or dec(nums)


class Solution:
    def isMonotonic(self, A: List[int]) -> bool:
        return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or all(A[i] >= A[i + 1] for i in range(len(A) - 1))
            
# may be used when we need to detect peak + or -: 
class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        lenght = len(arr)
        if lenght < 3:
            return False
        i = 0
        while i+1 < lenght and arr[i] < arr[i+1]:
            i+=1

        if i == lenght-1 or i == 0:
            return False

        while i+1 < lenght and arr[i] > arr[i+1]:
            i+=1
        
        return i == lenght-1
    

# very MAYBE??? the biggest difference in the timeseries: 
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        minn = float('inf')
        for prc in prices:
            minn = min(prc, minn)
            temp = prc - minn
            ans = max(temp, ans)
        return ans


# ________ worked part
from collections import Counter, defaultdict
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        dct = Counter()
        ans = 0
        for num in nums:
            ans += dct[num]
            dct[num] += 1
        return ans


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = defaultdict(list)
        for word in strs:
            key = ''.join(sorted(word))
            ans[key].append(word)


        return list(ans.values())
    
class Solution(object):
    def digitCount(self, num):

        lst = [0] * 10
        for n in num:
            lst[int(n)] += 1
        # print('lst', lst)
        
        for i in range(0, len(num)):
            if lst[i] != int(num[i]):
                
                return False
        return True
    
import re
class Solution:
    def isPalindrome(self, s: str) -> bool:

        # s = re.sub(r'[^a-zA-Z]', '', s)
        # s = re.sub(r'[0-9]', '', s)
        # s = ''.join([c.lower() for c in s if c.isalnum()])
        s = s.lower()
        l, r = 0, len(s)-1

        while l <= r:
            if not s[l].isalnum():
                l+= 1
                continue
            if not s[r].isalnum():
                r -= 1
                continue
            if s[l] == s[r]:
                l+= 1
                r-=1
            else:
                return False
        return True
    
    def firstPalindrome(self, words: List[str]) -> str:
        for w in words: 
            if self.isPalindrome(w):
                return w
            
        return ''



    

class Solution:
    def specialArray(self, nums: List[int]) -> int: # O(NlogN) and constant space
        nums.sort()
        lenght = len(nums)
        L, R = 0, lenght-1

        while L <= R:
            mid = (L + R) // 2

            if nums[mid] >= mid: 

                if lenght - nums[mid] == mid:
                    return mid
                else:
                    L = mid + 1
            else:
                R = mid-1

        return -1
        

# list node classes and funcs
import json
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

    def to_list(self):
        arr = []
        head = self
        while head:
            arr.append(head.val)
            head = head.next
        return arr

    def __repr__(self):
        arr = ''
        head = self
        while head:
            arr += str(head.val) + ' -> '
            head = head.next
        return arr

def stringToIntegerList(input):
    return json.loads(input)

def stringToListNode(input):
    # Generate list from the input
    if isinstance(input, list):
        input = str(input)
    numbers = stringToIntegerList(input)

    # Now convert that list into linked list
    dummyRoot = ListNode(0)
    ptr = dummyRoot
    for number in numbers:
        ptr.next = ListNode(number)
        ptr = ptr.next

    ptr = dummyRoot.next
    return ptr

def prettyPrintLinkedList(node):
    while node and node.next:
        print(str(node.val) + "->", end='')
        node = node.next

    if node:
        print(node.val)
    else:
        print("Empty LinkedList")

def main():
    import sys

    def readlines():
        for line in sys.stdin:
            yield line.strip('\n')

    lines = readlines()
    while True:
        try:
            line = next(lines)
            node = stringToListNode(line)
            prettyPrintLinkedList(node)
        except StopIteration:
            break



class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:

        hare = head
        turtle = head
        while hare and hare.next:

            turtle = turtle.next

            hare = hare.next.next

        return turtle
    
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1_dct = {key: -1 for key in nums1}
        stack = []
    
        for i in range(len(nums2)):
            while stack and nums2[i] > stack[-1]:
                if nums2[i] > stack[-1]:
                            
                    if stack[-1] in nums1_dct:
                        nums1_dct[stack[-1]] = nums2[i]
                    stack.pop()

            stack.append(i)


        return list(nums1_dct.values())
    

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        root = self
        if not root:
            return '[]'
        output = []
        queue = [root]
        current = 0
        while current != len(queue):
            node = queue[current]
            current = current + 1

            if not node:
                output.append('null')
                continue

            output.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
        while output[-1] == 'null':
            output.pop()
        return '[' + ', '.join(output) + ']'

def treeNodeToString(root):
    if not root:
        return []
    output = []
    queue = [root]
    current = 0
    while current != len(queue):
        node = queue[current]
        current = current + 1

        if not node:
            output.append('null')
            continue

        output.append(node.val)
        queue.append(node.left)
        queue.append(node.right)
    while output[-1] == 'null':
        output.pop()
    return output

def stringToTreeNode(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(',')]
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root

def prettyPrintTree(node, prefix="", isLeft=True):
    if not node:
        print("Empty Tree")
        return

    if node.right:
        prettyPrintTree(node.right, prefix + ("│   " if isLeft else "    "), False)

    print(prefix + ("└── " if isLeft else "┌── ") + str(node.val))

    if node.left:
        prettyPrintTree(node.left, prefix + ("    " if isLeft else "│   "), True)

def main_tree():
    import sys

    def readlines():
        for line in sys.stdin:
            yield line.strip('\n')

    lines = readlines()
    while True:
        try:
            line = next(lines)
            node = stringToTreeNode(line)
            prettyPrintTree(node)
        except StopIteration:
            break

def binary_tree_traverse(root):
    arr = []
    stack = [root]
    while stack:
        cur = stack.pop()
        arr.append(cur.val)
        if cur.right:
            stack.append(cur.left)
        if cur.left:
            stack.append(cur.right)
    return arr


    
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        stack = [(root, targetSum)]

        while stack:
            cur, cur_sum = stack.pop()
            cur_sum -= cur.val

            if not cur.left and not cur.right and not cur_sum:
                return True
            
            if cur.right:
                stack.append((cur.right, cur_sum))
            if cur.left:
                stack.append((cur.left, cur_sum))
            
            

        return False
            




if __name__ == "__main__":

    # root = TreeNode(1)
    # root.left = TreeNode(2)
    # root.right = TreeNode(3)
    # # print(root.val, root.left.val, root.right.val)
    # # prettyPrintTree(root)
    # # print(root)
    # print(binary_tree_traverse(root))

    # print()

    root = stringToTreeNode('[5,4,8,11,null,13,4,7,2,null,null,null,1]')
    # prettyPrintTree(root)
    # print(root)
    # print(binary_tree_traverse(root))]
    target = 22

    print(Solution().hasPathSum(root = root, targetSum=target))

    # main()
    # head = [1,2,3,4,5]
    # head = stringToListNode(head)
    
    # prettyPrintLinkedList(head)
    # sol = Solution()
    # print('answer is: \n', sol.reverseList(head).to_list())


    # sol = Solution()
    # nums1 = [4,1,2,0]
    # nums2 = [3,4,2,0,1]
    # print(sol.nextGreaterElement(nums1, nums2))

    # print(head.val, head.next.next.val)

    # print(head)
    # print(head.to_list())

    


    # # import openai

    # # print(openai.__version__)
    # sol = Solution()
    # nums = [3,5]
    # # target = 6
    # print(sol.specialArray(nums))
