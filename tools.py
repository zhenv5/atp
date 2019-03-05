import os.path
def sprint(dir_name,file_name,line):
	print line
	with open(os.path.join(dir_name,file_name),"a") as f:
		f.write("%s \n" % line)

def dir_tail_name(file_name):
	import os.path
	dir_name = os.path.dirname(os.path.abspath(file_name))
	#dir_name = os.path.dirname(file_name)
	head, tail = os.path.split(file_name)
	print("dir name: %s, file_name: %s" % (dir_name,tail))
	return dir_name,tail

def run_command(command,is_print = True):
	import subprocess
	print("command: %s" % command)
	p = subprocess.Popen(command,shell = True, stdout = subprocess.PIPE)
	o = p.communicate() 
	if is_print:
		print o[0]