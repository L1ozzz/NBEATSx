from web3 import Web3
import zlib
import ast

# 设置 Web3 连接
w3 = Web3(Web3.HTTPProvider('https://sepolia.infura.io/v3/4d90f4e7d85e431aa40db5e8ee64105a'))

# 合约地址和 ABI
contract_address = '0xd9145CCE52D386f254917e481eB44e9943F39138'  # 替换为您的合约地址
contract_abi = [
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "owner",
				"type": "address"
			}
		],
		"name": "OwnableInvalidOwner",
		"type": "error"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "account",
				"type": "address"
			}
		],
		"name": "OwnableUnauthorizedAccount",
		"type": "error"
	},
	{
		"anonymous": False,
		"inputs": [
			{
				"indexed": True,
				"internalType": "uint256",
				"name": "date",
				"type": "uint256"
			},
			{
				"indexed": False,
				"internalType": "bytes",
				"name": "compressedData",
				"type": "bytes"
			}
		],
		"name": "DataStored",
		"type": "event"
	},
	{
		"anonymous": False,
		"inputs": [
			{
				"indexed": True,
				"internalType": "address",
				"name": "previousOwner",
				"type": "address"
			},
			{
				"indexed": True,
				"internalType": "address",
				"name": "newOwner",
				"type": "address"
			}
		],
		"name": "OwnershipTransferred",
		"type": "event"
	},
	{
		"inputs": [],
		"name": "renounceOwnership",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "date",
				"type": "uint256"
			},
			{
				"internalType": "bytes",
				"name": "compressedData",
				"type": "bytes"
			}
		],
		"name": "storeData",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "newOwner",
				"type": "address"
			}
		],
		"name": "transferOwnership",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "date",
				"type": "uint256"
			}
		],
		"name": "dataExists",
		"outputs": [
			{
				"internalType": "bool",
				"name": "",
				"type": "bool"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "date",
				"type": "uint256"
			}
		],
		"name": "getData",
		"outputs": [
			{
				"internalType": "bytes",
				"name": "",
				"type": "bytes"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "owner",
		"outputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"stateMutability": "view",
		"type": "function"
	}
]

# 创建合约实例
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# 交易哈希
tx_hash = '0xcfa33936693576d8e051af34c06982ab1ffc627c1119c5f32b53979c8e954718'  # 替换为您的交易哈希

# 获取交易详情
try:
    tx = w3.eth.get_transaction(tx_hash)
    print(tx)
except Exception as e:
    print(f"Error fetching transaction: {e}")
    exit()

# 获取 Input Data
input_data = tx.input
print(input_data)

# 解码 Input Data
try:
    function_name, function_params = contract.decode_function_input(input_data)
    print(f"Function called: {function_name.fn_name}")
    print(f"Function parameters: {function_params}")
except Exception as e:
    print(f"Error decoding input data: {e}")
    exit()

datetimm = function_params['date']

print(type(datetimm))
# 获取压缩数据
compressed_data = function_params['compressedData']

# 解压缩数据
try:
    decompressed_data = zlib.decompress(compressed_data)
    print(decompressed_data)
except Exception as e:
    print(f"Error decompressing data: {e}")
    exit()

# 将字节数据解码为字符串
data_str = decompressed_data.decode('utf-8')

# 分割数据字符串
first_comma_index = data_str.find(',')
last_comma_index = data_str.rfind(',')

# 检查格式
if first_comma_index == -1 or last_comma_index == -1 or first_comma_index == last_comma_index:
    print("Data string format is incorrect.")
    exit()

# 提取各个部分
date_str = data_str[:first_comma_index]
fields_str = data_str[first_comma_index + 1:last_comma_index]
season = data_str[last_comma_index + 1:]

print(f"Date: {date_str}")
print(f"Fields string: {fields_str}")
print(f"Season: {season}")

# 处理 fields_str
fields_str = fields_str.strip('[]')  # 移除中括号
fields_list_str = fields_str.split(',')  # 按逗号分割

# 将字符串转换为浮点数，并除以 100
try:
    fields = [float(num.strip()) / 100 for num in fields_list_str]
except Exception as e:
    print(f"Error processing fields: {e}")
    exit()

print(f"Fields: {fields}")
'''
owner_address = contract.functions.owner().call()
print(f"Contract owner: {owner_address}")
'''
date_to_query = 20240522  # 替换为您存储数据时使用的日期
print(f"Date to query: {date_to_query}, Type: {type(date_to_query)}")


# 检查数据是否存在
try:
    data_exists = contract.functions.dataExists(date_to_query).call()
    print(f"Data exists for date {date_to_query}: {data_exists}")
    if data_exists:
        compressed_data = contract.functions.getData(date_to_query).call()
        print(f"Compressed data: {compressed_data}")
    else:
        print(f"No data found for date {date_to_query}")
        exit()
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()





