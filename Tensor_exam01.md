

```python
## practice tensorflow
### constant()
#### : 3.0, 4.0의 정수 값을 가지는 노드와 operation을 만든다. : graph를 만든다.
#### : session을 만들어서 sess.run으로 실행 후 업데이트 또는 결과를 리턴
```


```python
import tensorflow as tf
# build graph
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # tf.float32가 암묵적으로 들어감
node3 = tf.add(node1, node2) # same as node3 = node1 + node2

#sess.run() execute and return value or update
with tf.Session() as sess:
    print("sess.run(node1, node2): ", sess.run([node1,node2]))
    print("sess.run(node3) : ", sess.run(node3))
```


```python
### placeholder()
#### placeholder : placeholder란 노드를 만든다. * 값이 들어있지 않음
#### feed_dict를 이용해서 값을 넣어줌.
#### shape를 정해주지 않으면 default : None, None 
#### None : 아무 크기나 가능
```


```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node, feed_dict={a:3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b: [2,4]}))
```


```python
### Tensor
#### rank : 몇 차원으로 되어있냐
#### shape : 각 element에 몇 개가 들어있냐
#### datatype: 대부분의 경우 flaot32 or int32
```
