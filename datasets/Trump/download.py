import os
from subprocess import check_output
from math import floor

def get_duration(path):
	duration = str(check_output('/home/a/addityap/ffmpeg/ffmpeg-git-20190521-amd64-static/ffprobe -i  '+ path + ' 2>&1 | grep "Duration"', shell=True))
	duration = duration.split(",")[0].split("Duration:")[1].strip()
	h, m, s = duration.split(':')
	duration = int(h) * 3600 + int(m) * 60 + round(float(s))
	return duration

counter = 0
with open('trump_addresses.txt') as f:
	for cur_link in f:
		counter = counter + 1
		try:
			link = cur_link.rstrip('\n').strip()
			save_f = 'raw_new/' + str(counter) + '.mp4'
			command = '/home/a/addityap/youtube-dl -f 22 "' + link + '" --output ' + save_f + ' --min-filesize 2m --no-playlist'
			print(command)
			# os.system(command)
			print('Downloaded: ', link, save_f)
			duration = get_duration(save_f)
			cut_f = 'cut_new/' + str(counter) + '.mp4'
			new_duration = duration - 40
			cut_command = '/home/a/addityap/ffmpeg/ffmpeg-git-20190521-amd64-static/ffmpeg -v quiet -i ' + save_f + ' -ss 30 -t ' + str(new_duration) + ' ' + cut_f
			os.system(cut_command)
			if new_duration != get_duration(cut_f):
				print("Error: ", cut_f, get_duration(cut_f), new_duration)
				os.system('rm ' + cut_f)

			c0_path = 'c0/' + str(counter) + '.mp4'
			c23_path = 'c23/' + str(counter) + '.mp4'
			c40_path = 'c40/' + str(counter) + '.mp4'
			os.system('cp ' + cut_f + ' ' + c0_path)
			os.system('/home/a/addityap/ffmpeg/ffmpeg-git-20190521-amd64-static/ffmpeg -v quiet -i ' + cut_f + ' -crf 23 ' + c23_path)
			os.system('/home/a/addityap/ffmpeg/ffmpeg-git-20190521-amd64-static/ffmpeg -v quiet -i ' + cut_f + ' -crf 40 ' + c40_path)

		except Exception as e:
			print("Error: ", str(counter) + '.mp4', e)