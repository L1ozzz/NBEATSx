import csv
from datetime import datetime
from web3 import Web3
import zlib
import time
import binascii

# 设置Web3连接
w3 = Web3(Web3.HTTPProvider('https://sepolia.infura.io/v3/4d90f4e7d85e431aa40db5e8ee64105a'))

# 合约地址和ABI
contract_address = '0xd9145CCE52D386f254917e481eB44e9943F39138'
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

# 设置账户
private_key = '85e6816ba9396d6ba3ab763a2dabc8dea26406da772fe5bd5f364bb021a90907'

account = w3.eth.account.from_key(private_key)

# 打印账户地址
print(f"Account address: {account.address}")

# 获取账户余额
balance = w3.eth.get_balance(account.address)
balance_eth = w3.from_wei(balance, 'gwei')
print(f"Account balance: {balance_eth} gwei")

# 获取最新区块号
latest_block_number = w3.eth.block_number
print(f"Latest block number: {latest_block_number}")

# 获取账户的已确认 Nonce
nonce = w3.eth.get_transaction_count(account.address, 'latest')
print(f"Current nonce (latest): {nonce}")

# Chain ID
chain_id = 11155111  # Sepolia 测试网的 Chain ID

# 设置 maxPriorityFeePerGas（矿工小费）
max_priority_fee = w3.to_wei(13, 'gwei')  # 您可以根据需要调整

# 读取数据集并处理
with open('data/test.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    nonce = w3.eth.get_transaction_count(account.address, 'latest')
    print(f"Current nonce (latest): {nonce}")
    for row in reader:
        # 日期转换
        date_str = row['Day(Local_Date)']
        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
        date = int(date_obj.strftime('%Y%m%d'))

        # 字段处理
        fields = []
        multiplier = 100  # 倍率

        # 列名称列表，按顺序排列
        field_names = [
            'GustDir(Deg)',
            'GustSpd(m/s)',
            'WindRun(Km)',
            'Rain(mm)',
            'Tmean(C)',
            'Tmax(C)',
            'Tmin(C)',
            'Tgmin(C)',
            'VapPress(hPa)',
            'ET10(C)',
            'Rad(MJ/m2)',
            'SoilM(%)'
        ]

        for field_name in field_names:
            value = float(row[field_name])
            value_int = int(value * multiplier)
            fields.append(value_int)

        # 季节
        season = row['Season']

        # 组合数据为字符串
        data_str = f"{date},{fields},{season}"
        data_bytes = data_str.encode('utf-8')

        # 压缩数据
        compressed_data = zlib.compress(data_bytes)
        print(f"Original size: {len(data_bytes)} bytes")
        print(f"Compressed size: {len(compressed_data)} bytes")

        # 获取当前 baseFee
        latest_block = w3.eth.get_block('latest')
        base_fee = latest_block['baseFeePerGas']
        print(f"Current base fee: {w3.from_wei(base_fee, 'gwei')} gwei")

        # 计算 maxFeePerGas
        max_fee = int(base_fee * 2) + max_priority_fee
        print(f"Calculated max fee per gas: {w3.from_wei(max_fee, 'gwei')} gwei")

        # 构建交易
        tx = contract.functions.storeData(date, compressed_data).build_transaction({
            'from': account.address,
            'nonce': nonce,
            'gas': 200000,  # 根据实际情况调整 gas 限制
            'maxFeePerGas': max_fee,
            'maxPriorityFeePerGas': max_priority_fee,
            'chainId': chain_id
        })

        # 打印交易详情
        print("Transaction details:")
        print(f"From: {tx['from']}")
        print(f"To: {tx['to']}")
        print(f"Nonce: {tx['nonce']}")
        print(f"Gas limit: {tx['gas']}")
        print(f"Max fee per gas: {w3.from_wei(tx['maxFeePerGas'], 'gwei')} gwei")
        print(f"Max priority fee per gas: {w3.from_wei(tx['maxPriorityFeePerGas'], 'gwei')} gwei")
        print(f"Chain ID: {tx['chainId']}")

        # 估计 Gas
        try:
            estimated_gas = w3.eth.estimate_gas(tx)
            print(f"Estimated gas: {estimated_gas}")
            tx['gas'] = estimated_gas + 50000  # 加一些余量
            print(f"Adjusted gas limit: {tx['gas']}")
        except Exception as e:
            print(f"Gas estimation failed: {e}")
            continue  # 跳过此交易

        # 计算预估的交易费用
        transaction_cost = tx['gas'] * tx['maxFeePerGas']
        transaction_cost_eth = w3.from_wei(transaction_cost, 'ether')
        print(f"Estimated transaction cost: {transaction_cost_eth} ETH")

        # 检查账户余额是否足够
        if balance < transaction_cost:
            print("Insufficient balance to cover transaction cost.")
            break  # 退出循环
        start_time = time.time()
        # 签名并发送交易
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
        try:
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            print(f"Transaction sent: {w3.to_hex(tx_hash)}")
        except Exception as e:
            print(f"Error sending transaction: {e}")
            continue  # 跳过此交易

        # 等待交易确认并解析事件
        try:
            print("Waiting for transaction receipt...")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)

            if receipt.status == 1:
                print('Transaction confirmed and data stored successfully.')
                end_time = time.time()
                transaction_time = end_time - start_time
                print(f"Transaction time: {transaction_time}")
                print(receipt)

                # 解析事件日志
                logs = contract.events.DataStored().process_receipt(receipt)
                print(logs)
                for event in logs:
                    event_args = event['args']
                    stored_date = event_args['date']
                    stored_compressed_data = event_args['compressedData']
                    print(f"Data stored with date: {stored_date}")
                    print(f"Compressed data (hex): {stored_compressed_data.hex()}")
            else:
                 print('Transaction failed.')
        except Exception as e:
            print(f'Waiting for transaction receipt failed: {e}')
            # 检查交易是否存在于本地节点
            try:
                transaction = w3.eth.get_transaction(tx_hash)
                print("Transaction found in local node:")
                print(transaction)
            except Exception as tx_error:
                print(f"Transaction not found in local node: {tx_error}")
            continue  # 跳过此交易

        # 更新账户余额
        balance = w3.eth.get_balance(account.address)
        balance_eth = w3.from_wei(balance, 'gwei')
        print(f"Updated account balance: {balance_eth} gwei")

        # 添加分隔符
        print("-" * 50)
'''
account = w3.eth.account.from_key(private_key)

# 获取当前 Nonce，包括未确认的交易
nonce = w3.eth.get_transaction_count(account.address, 'pending')

# Chain ID
chain_id = 11155111  # Sepolia 测试网的 Chain ID

# 获取当前 baseFee
latest_block = w3.eth.get_block('latest')
base_fee = latest_block['baseFeePerGas']
print(f"Current base fee: {w3.from_wei(base_fee, 'gwei')} gwei")

# 设置 maxPriorityFeePerGas（矿工小费）
max_priority_fee = w3.to_wei(2, 'gwei')  # 您可以根据需要调整

# 计算 maxFeePerGas
max_fee = base_fee + max_priority_fee

# 读取数据集并处理
with open('data/test.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 日期转换
        date_str = row['Day(Local_Date)']
        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
        date = int(date_obj.strftime('%Y%m%d'))

        # 字段处理
        fields = []
        multiplier = 100  # 倍率

        # 列名称列表，按顺序排列
        field_names = [
            'GustDir(Deg)',
            'GustSpd(m/s)',
            'WindRun(Km)',
            'Rain(mm)',
            'Tmean(C)',
            'Tmax(C)',
            'Tmin(C)',
            'Tgmin(C)',
            'VapPress(hPa)',
            'ET10(C)',
            'Rad(MJ/m2)',
            'SoilM(%)'
        ]

        for field_name in field_names:
            value = float(row[field_name])
            value_int = int(value * multiplier)
            fields.append(value_int)

        # 季节
        season = row['Season']

        # 构建交易
        tx = contract.functions.storeData(date, fields, season).build_transaction({
            'from': account.address,
            'nonce': nonce,
            'gas': 500000,  # 初始 Gas 限制
            'maxFeePerGas': max_fee,
            'maxPriorityFeePerGas': max_priority_fee,
            'chainId': chain_id
        })

        # 估计 Gas
        try:
            estimated_gas = w3.eth.estimate_gas(tx)
            print(f'Estimated gas: {estimated_gas}')
            tx['gas'] = estimated_gas + 100000  # 加一些余量
        except Exception as e:
            print(f'Gas estimation failed: {e}')
            continue  # 跳过此交易

        # 签名并发送交易
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
        try:
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            print(f'Transaction sent: {w3.to_hex(tx_hash)}')
        except Exception as e:
            print(f'Error sending transaction: {e}')
            continue  # 跳过此交易

        # 等待交易确认（增加超时时间并添加异常处理）
        try:
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            if receipt.status == 1:
                print('Transaction confirmed and data stored successfully.')
            else:
                print('Transaction failed.')
        except Exception as e:
            print(f'Waiting for transaction receipt failed: {e}')
            continue  # 跳过此交易

        # 增加 Nonce 值
        nonce = nonce + 1
'''