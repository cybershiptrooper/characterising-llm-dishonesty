
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Modular Arithmatic')
#     parser.add_argument('-n', '--n', type=int, required=False, help='number of samples', default=15)
#     parser.add_argument('-m', '--m', type=int, required=False, help='modulus', default=10)
#     parser.add_argument('-max', '--max', type=int, required=False, help='max value of samples', default=20)
#     parser.add_argument('-s', '--s', type=int, required=False, help='seed', default=0)
#     parser.add_argument('-test', '--test', action='store_true', help='test mode')
#     parser.add_argument('-p', '--p', type=float, required=False, help='false probability', default=0.5)

#     args = parser.parse_args()
#     n = args.n
#     m = args.m
#     maxinum = args.max
#     dataset = make_data(n, args.p, m, maxinum, args.test, args.s)
#     print(dataset.sample(n))
