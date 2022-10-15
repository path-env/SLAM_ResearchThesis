str = "The times table";
print(str);
print(len(str) * "=");

for i in range(1, 13):
  for j in range(1, 13):
    print("%2d x %2d = %3d" % (i, j, i * j));
  print("-" * 13);