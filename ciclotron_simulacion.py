import math as math
import numpy as np
import matplotlib.pyplot as plt

class Particle:
	def __init__(self, pos, vel, mass, charge):
		self.pos = pos
		self.vel = vel
		self.mass = mass
		self.charge = charge

m=1.67E-27
e=+1.60E-19
B=2.5
E=5000000.0
speed=0.05


speed_of_light = 3.0E08
proton = Particle([0.00, 0.0, 0.0], [speed*speed_of_light, 0.0, 0.0], 1.67E-27, +1.60E-19)
e_field = [E, 0.0, 0.0]
magn_uniform_field = [0.0, 0.0, -B]

b = np.linalg.norm(magn_uniform_field)
v_i = np.linalg.norm(proton.vel)
expected_radius = v_i / ( b * (proton.charge/proton.mass ) )
spacing = 0.5 * expected_radius

expected_period = 2.0*math.pi/( b * (proton.charge/proton.mass ) )
number_of_points = 400*2
guess_for_delta_t = expected_period / number_of_points
count = 0
jumps =0
jumps_max = 51

def EM( q_over_m, position, velocity ):
	global count, jumps
	if jumps >= jumps_max:
		a = 0
	else:
		if position[0] >= 0 or position[0] <= -spacing:
			a = np.cross(velocity, magn_uniform_field)
			a = a * q_over_m
			if count:
				count = 0
				jumps = jumps + 1
		else:
			a = np.array(e_field)
			a = a * q_over_m
			if position[1] > expected_radius:
				a = -a
			count = count + 1

	return a

def runge_kutta_4(particle, max_iter, delta_t):
	q_over_m = particle.charge / particle.mass
	results = []
	vex=[]
	vey=[]
	i = 0
	p0 = np.array(particle.pos)
	v0 = np.array(particle.vel)

	for i in range(max_iter):
		i += 1
		p1 = p0
		v1 = v0
		a1 = delta_t * EM( q_over_m, p1, v1 )
		v1 = delta_t * v1
		
		p2 = p0 + (v1 * 0.5)
		v2 = v0 + (a1 * 0.5)
		a2 = delta_t * EM( q_over_m, p2, v2 )
		v2 = delta_t * v2
		
		p3 = p0 + (v2 * 0.5)
		v3 = v0 + (a2 * 0.5)
		a3 = delta_t * EM( q_over_m, p3, v3 )
		v3 = delta_t * v3
		
		p4 = p0 + v3
		v4 = v0 + a3
		a4 = delta_t * EM( q_over_m, p4, v4 )
		v4 = delta_t * v4
		
		dv = (a1 + 2.0 * (a2 + a3) + a4)
		v0 = v0 + dv / 6.0
		
		dp = (v1 + 2.0 * (v2 + v3) + v4)
		p0 = p0 + dp / 6.0
		
		if p0[0] >= -10.0 * expected_radius and p0[0] <= 10.0 * expected_radius and \
                   p0[1] >= -10.0 * expected_radius and p0[1] <= 10.0 * expected_radius :
			p0[2] = np.linalg.norm(v0)
			vex.append(v0[0])
			vey.append(v0[1])
			results.append(p0.copy())
			p0[2] = 0.0

	return results,vex,vey
	

def plots(particle, max_iter):

	x = []
	y = []

	x.append(particle.pos[0])
	y.append(particle.pos[1])


	z = []
	v0 = np.linalg.norm(particle.vel)
	z.append(v0)
	delta_t = guess_for_delta_t
	results,Vx,Vy = runge_kutta_4(particle, max_iter, delta_t)
	r=np.asarray(results)	
	X,Y=r[:,0],r[:,1]
	xc = []
	yc = []
	x = []
	y = []
	for p in results:
		z.append(p[2])
		if p[0] >= 0 or p[0] <= -spacing :
			if len(xc) :
				plt.plot(xc,yc, color='green',linewidth=0.95)
				xc = []
				yc = []
			x.append(p[0])
			y.append(p[1])
		else:
			if len(xc) :
				plt.plot(x,y, color='green',linewidth=0.95)
				x = []
				y = []
			xc.append(p[0])
			yc.append(p[1])
	plt.title("Simulación del Ciclotrón")
	plt.xlabel(r"$X$ $[m]$")
	plt.ylabel(r"$Y$ $[m]$")
	plt.savefig('ciclotron.jpeg')

	return X,Y,Vx,Vy


X,Y,VX,VY=plots(proton, 20000)

plt.close()

plt.figure(figsize=(8,5))

VX=np.gradient(X,2)
plt.plot(X,VX)
plt.title(r"Simulación del Ciclotrón Diagrama $V_{X}$ vs $X$")
plt.ylabel(r"$V_{X}$ $[m/s]$")
plt.xlabel(r"$X$ $[m]$")
plt.savefig('ciclotron_vx_x.jpeg')
plt.close()

plt.figure(figsize=(8,5))

VY=np.gradient(Y,2)
plt.plot(Y,VY)
plt.title(r"Simulación del Ciclotrón Diagrama $V_{Y}$ vs $Y$")
plt.ylabel(r"$V_{Y}$ $[m/s]$")
plt.xlabel(r"$Y$ $[m]$")
plt.savefig('ciclotron_vy_y.jpeg')
plt.close()

plt.figure(figsize=(8,5))

T=(2*np.pi*(X**2+Y**2)**(0.5))/(VX**2+VY**2)**(1/2)
t=np.linspace(0,len(T),len(T))
plt.plot(t,T)
plt.title(r"Simulación del Ciclotrón Diagrama $T$ vs $t$")
plt.xlabel(r"$t$ $[s]$")
plt.ylabel(r"$T$ $[s]$")
plt.savefig('ciclotron_T_t.jpeg')
plt.close()

m=1.67E-27
e=+1.60E-19
B=1.5

plt.figure(figsize=(8,5))

r=(X**2+Y**2)**(1/2)
R=(m*((VX**2+VY**2)**(1/2)))/(e*B)
plt.plot(R,r)
plt.title(r"Diagrama $r$ $(p)$ vs $r$ $(x,y)$")
plt.xlabel(r"$r$ $(x,y)$ $[m]$")
plt.ylabel(r"$r$ $(p)$ $[m]$")
plt.savefig('ciclotron_r.jpeg')
plt.close()



