class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        return x * y

    def backward(self,dout):
        dx = self.y * dout
        dy = self.x * dout

        return dx, dy


apple = 100
apple_num = 2
tax = 1.1

apple_network = MulLayer()
tx_network = MulLayer()
apple_price = apple_network.forward(apple, apple_num)
price = tx_network.forward(tax, apple_price)

print(apple_price)
print(price)

dprice = 1
dtax, d_apple_price = tx_network.backward(dprice)
dapple, dapple_num = apple_network.backward(d_apple_price)

print(dapple, dapple_num, dtax)


class Add_Layer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        return x+y

    def backward(self,dout):
        return dout, dout

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
mul_orange_layer = MulLayer()
add_fruit_layer = Add_Layer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
total_price_no_tax = add_fruit_layer.forward(apple_price, orange_price)
total_price = mul_tax_layer.forward(total_price_no_tax, tax)

print("apple price : %d"%apple_price)
print("orange price : %d"%orange_price)
print("total price no tax : %d"%total_price_no_tax)
print("total price : %f"%total_price)

# backward
dtotal_price = 1
dtotal_price_no_tax, dtax = mul_tax_layer.backward(dtotal_price)
dapple_price, dorange_price = add_fruit_layer.backward(dtotal_price_no_tax)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

