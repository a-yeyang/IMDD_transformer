import java.util.*;

public class LRUCache {
    // 定义双向链表节点
    class Node {
        int key;
        int value;
        Node prev;
        Node next;
        public Node() {}
        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private Map<Integer, Node> cache = new HashMap<>();
    private int size;
    private int capacity;
    private Node head, tail;

    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        // 使用伪头部和伪尾部节点，避免处理空指针的边界问题
        head = new Node();
        tail = new Node();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        Node node = cache.get(key);
        if (node == null) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        moveToHead(node);
        return node.value;
    }

    public void put(int key, int value) {
        Node node = cache.get(key);
        if (node == null) {
            // 如果 key 不存在，创建一个新的节点
            Node newNode = new Node(key, value);
            cache.put(key, newNode);
            addToHead(newNode);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                Node res = removeTail();
                cache.remove(res.key);
                --size;
            }
        } else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node.value = value;
            moveToHead(node);
        }
    }

    private void addToHead(Node node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void moveToHead(Node node) {
        removeNode(node);
        addToHead(node);
    }

    private Node removeTail() {
        Node res = tail.prev;
        removeNode(res);
        return res;
    }

    // --- Main 函数用于测试 ---
    public static void main(String[] args) {
        LRUCache lru = new LRUCache(2); // 容量为 2

        lru.put(1, 1); // 缓存是 {1=1}
        lru.put(2, 2); // 缓存是 {1=1, 2=2}
        System.out.println("Get 1: " + lru.get(1));    // 返回 1，缓存是 {2=2, 1=1}
        
        lru.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
        System.out.println("Get 2 (should be -1): " + lru.get(2)); 
        
        lru.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {3=3, 4=4}
        System.out.println("Get 1 (should be -1): " + lru.get(1));
        System.out.println("Get 3: " + lru.get(3));    // 返回 3
        System.out.println("Get 4: " + lru.get(4));    // 返回 4
    }
}