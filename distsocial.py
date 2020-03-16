# -*- coding: utf-8 -*-
"""Influence of social distancing/isolation on disease spread simulation.

This script demonstrates the influence social distancing can have on the spread
of a disease such as Covid-19 in a population.

Originally based on A washingtonpost artice from March 2020.
Forked from: https://github.com/tioguerra/distancia_social
Adapted by: https://github.com/jvs--

Todo:
    * Fix distancing vs isolation
    * Try in larger population
    * Check survial rate with silent vectors
    * Fix units and hard coded number to actually corespond to something
        - eg.: Sickness length seems to be update cycles.
    * Add health care system capacity
    * Move vectors to a new location (air travel)
    * People movement is colision based, maybe better some kind of random walk、
        or around a central location (home, work). Can be spawning point
"""
import numpy as np
import pygame
from pygame import locals

# Setup
# Background of the playground
BACKGROUND_COLOR = pygame.Color('#f0f0f0')
# Background of the graph
GRAPH_BACKGROUND_COLOR = pygame.Color('#FFFFFF')
# Colors for the graph
SICK_COLOR = pygame.Color('#FF606B')
CURED_COLOR = pygame.Color('#ACF39D')
DEAD_COLOR = pygame.Color('#000000')
HEALTHY_COLOR = pygame.Color('#B6D9DA')

# Possible status of a person
HEALTHY, SICK, CURED, DEAD = range(4)

# Chance of dying from the disease if infected (not overall)
DIE_CHANCE = 0.04
# TODO: a 100 what? this is updates cylces but what does it mean in real units
SICKNESS_LENGTH = 50
# This adjusts the width of the graph (the smaller, the wider)
TIME_SCALE = 1.0

# Modulus of the velocity of each moving person
VELOCITY = 10.0

# Modify here to simulate social distancing
# (this tells how many people will be stationary,
#  for instance, 0.7 means 70% will not move(?) be isolated at home)
# SOCIAL_DISTANCING = 0.0
SOCIAL_DISTANCING = 0.2
SOCIAL_ISOLATION = 0.0  # Complete isolation
SOCIAL_DISTANCING_MASS = 1000.0
TRAVEL = 0.5

# Size of the population
POPULATION = 1000


class Person():
    ''' Class to hold a person status, position and velocity
    '''
    status = HEALTHY

    def __init__(self, pos, vel, sd, si):
        self.pos = pos
        self.vel = vel
        self.social_distancing = sd
        self.social_isolation = si
        self.counter = 0
        if self.social_distancing:
            self.mass = SOCIAL_DISTANCING_MASS
        else:
            self.mass = 1.0


class Sim():
    ''' This is the simulation class. Most of the code
        is in here
    '''

    def __init__(self, pop=POPULATION, w=1700, h=500, g=400, r=13):
        # TODO: set window size relative to screen, not hard coded
        # Simulation Parameters
        self.width = w
        self.height = h
        self.radius = r
        self.graph = g
        # Generate the initial population
        self.createPopulation(pop)
        # Init PyGame stuff
        pygame.init()
        # Create the window
        self.screen = pygame.display.set_mode(
            (self.width, self.height+self.graph))
        # Load the images
        self.healthy_img = pygame.image.load('healthy.png')
        self.sick_img = pygame.image.load('sick.png')
        self.cured_img = pygame.image.load('cured.png')
        self.dead_img = pygame.image.load('dead.png')
        # Initialize counters
        self.sick = self.cured = self.dead = 0
        self.counter = 0
        # Simulation starts paused. Press space to start.
        self.start = False

    def checkPopulationCollision(self, person):
        # Check for colisions on the initial population
        for other in self.people:
            if other != person:
                if self.checkCollision(person, other):
                    return True
        return False

    def checkPopulationBoundaries(self):
        # Check the "walls" of the playgraound
        # This should be checked smarter
        for person in self.people:
            if (person.pos[0] < self.radius and person.vel[0] < 0) \
                    or (person.pos[0] > self.width - self.radius and person.vel[0] > 0):
                person.vel[0] = -person.vel[0]
                person.pos[0] += person.vel[0]
            if (person.pos[1] < self.radius and person.vel[1] < 0) \
                    or (person.pos[1] > self.height - self.radius and person.vel[1]) > 0:
                person.vel[1] = -person.vel[1]
                person.pos[1] += person.vel[1]

    def checkCollision(self, person1, person2):
        # Verify if one person touches another
        if person1 != person2:
            dist = np.linalg.norm(person1.pos - person2.pos)
            # if dist <= 2.0*self.radius:
            if dist <= 1.0*self.radius:  # I think this is just for drawing so
                return True
        return False

    def randomVelocity(self):
        # Generates an initial random velocity
        n = 0
        while n == 0:
            v = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            n = np.linalg.norm(v)
        return VELOCITY * v / n

    def createPopulation(self, pop):
        # Generates the initial population
        self.people = []
        while len(self.people) < pop:
            # Uniform distribution of positions
            pos = np.array([np.random.uniform(self.radius, self.width-self.radius),
                            np.random.uniform(self.radius, self.height-self.radius)])
            # Uniform distribution of velocities
            vel = self.randomVelocity()
            # Make sure some people have the necessary social distancing
            social_distancing = False
            social_isolation = False
            if np.random.rand() < SOCIAL_DISTANCING:
                vel = np.array([0.0, 0.0])
                social_distancing = True
            if np.random.rand() < SOCIAL_ISOLATION:
                vel = np.array([0.0, 0.0])
                social_distancing = True
                social_isolation = True
            # Creates a person
            p = Person(pos, vel, social_distancing, social_isolation)
            # Avoid creating people on top of each other
            if not self.checkPopulationCollision(p):
                self.people.append(p)
        # Randomly chooses someone to be sick
        np.random.choice(self.people).status = SICK

    def drawPopulation(self):
        # Draws the population on the screen
        pygame.draw.rect(self.screen, BACKGROUND_COLOR,
                         (0, 0, self.width, self.height))
        for person in self.people:
            # This is to adjust for the images size when using images instead
            # of dots so remove for now:
            # print(person.pos, np.array([self.radius, self.radius]))
            pos = person.pos - np.array([self.radius, self.radius])
            x = int(pos[0])
            y = int(pos[1])
            if person.status == HEALTHY:
                pygame.draw.circle(self.screen, HEALTHY_COLOR, (x, y), 2)
                # If you want images instead:
                # self.screen.blit(self.healthy_img, (x, y))
            elif person.status == SICK:
                pygame.draw.circle(self.screen, SICK_COLOR, (x, y), 2)
                # self.screen.blit(self.sick_img, (x, y))
            elif person.status == CURED:
                pygame.draw.circle(self.screen, CURED_COLOR, (x, y), 2)
                # self.screen.blit(self.cured_img, (x, y))
            elif person.status == DEAD:
                pygame.draw.circle(self.screen, DEAD_COLOR, (x, y), 2)
                # self.screen.blit(self.dead_img, (x, y))

    def drawGraph(self):
        # TODO: Move this out of pygame into plt or something
        # This will plot the cumulative cases in a graph at the bottom of the sim window
        counter = int(self.counter*TIME_SCALE)
        pygame.draw.rect(self.screen, GRAPH_BACKGROUND_COLOR,
                         (counter, self.height, self.width-counter, self.graph))
        sick = (self.sick / float(len(self.people))) * self.graph
        pygame.draw.line(self.screen, SICK_COLOR, (counter, self.height + self.graph),
                         (counter, self.height + self.graph - sick))
        dead = (self.dead / float(len(self.people))) * self.graph
        pygame.draw.line(self.screen, DEAD_COLOR, (counter, self.height),
                         (counter, self.height + dead))
        cured = (self.cured / float(len(self.people))) * self.graph
        pygame.draw.line(self.screen, CURED_COLOR, (counter, self.height + dead + cured),
                         (counter, self.height + dead))
        pygame.draw.line(self.screen, HEALTHY_COLOR, (counter, self.height + self.graph - sick),
                         (counter, self.height + dead + cured))

    def update(self):
        # Main function to update all the positions and "infect" people
        for i in range(len(self.people)):
            # If someone is not dead, it will move
            if (self.people[i].status != DEAD or not(self.people[j].social_isolation)):
                # Update position
                if not self.people[i].social_distancing:
                    self.people[i].pos += self.people[i].vel
                # If the person is sick, the counter will increment
                if self.people[i].status == SICK:
                    self.people[i].counter += 1
                    # When it reaches 300, the person heals or dies
                    # Todo: Make this counter mean a real time measuer
                    if self.people[i].counter == SICKNESS_LENGTH:
                        self.sick -= 1
                        if np.random.rand() > DIE_CHANCE:
                            self.people[i].status = CURED
                            self.cured += 1
                        else:
                            self.people[i].status = DEAD
                            self.dead += 1
                # Check collisions (only for moving people)
                if not self.people[i].social_distancing:
                    for j in range(0, len(self.people)):
                        # Can only collide with someone not dead
                        if (self.people[j].status != DEAD or self.people[j].social_isolation) and i != j:
                            # If there is a colision
                            if self.checkCollision(self.people[i], self.people[j]):
                                # Check if colition is with a socially distanced person
                                if self.people[j].social_distancing:
                                    # If the person is socially distanced, only the "colider" changes direction
                                    v1 = self.people[i].vel
                                    v2 = self.people[j].vel
                                    x1 = self.people[i].pos
                                    x2 = self.people[j].pos
                                    v1_ = v1 - 2.0 * \
                                        (x1-x2) * np.dot(v1-v2, x1-x2) / \
                                        2.0*self.radius
                                    if np.linalg.norm(v1_) > 0.0:
                                        v1_ = 2.0 * v1_ / np.linalg.norm(v1_)
                                    self.people[i].vel = v1_
                                    self.people[i].pos += self.people[i].vel
                                else:
                                    # Elastic colision, agents exchange their velocities (if not stationary)
                                    self.people[i].vel, self.people[j].vel = self.people[j].vel, self.people[i].vel
                                # If one of the two was sick, the other will be sick as well
                                if self.people[i].status == SICK:
                                    if self.people[j].status == HEALTHY:
                                        self.people[j].status = SICK
                                        self.people[j].counter = 0
                                        self.sick += 1
                                elif self.people[i].status == HEALTHY:
                                    if self.people[j].status == SICK:
                                        self.people[i].status = SICK
                                        self.people[i].counter = 0
                                        self.sick += 1
        if np.random.rand() < TRAVEL:
            # TODO: Do this less bad
            # Pick a random person (doesn't test if the person is isolated etc yet)
            index = int(np.random.randint(0, len(self.people)))
            # Choose a travel destination
            # Needs to be float because we got the vector calculation in float
            pos = np.array([float(np.random.randint(low=5, high=self.width-5)),
                            float(np.random.randint(low=5, high=self.height-5))])
            # print(pos)
            # print(self.people[index].pos)
            self.people[index].pos = pos
        # Do not let people wander outside the bounding box of the window
        self.checkPopulationBoundaries()
        # Simulation counter
        self.counter += 1

    def run(self):
        # This function runs the simulation
        quit = False
        while not quit:
            # Checks if the spacebar was pressed already or not
            # (after pressing spacebar, self.start will be True)
            if self.start:
                # Update simulation
                self.update()
                # Draws agents
                # TODO: Silently run simulation instead of drawing all steps?
                #       Right now you can barely run N = 1000
                #       Check if drawing or colision producing the bottleneck
                self.drawPopulation()
                # Draw graph
                self.drawGraph()
                # Refreshes the actual image on the screen
                pygame.display.flip()
            # Check for quit or spacebar presses
            for event in pygame.event.get():
                # TODO: This looks weird to me... fix
                if event.type == pygame.QUIT:
                    quit = True
                elif event.type == pygame.KEYUP:
                    # TODO: Display start by PRESSING SPACE or just start
                    if event.key == locals.K_SPACE:
                        self.start = True


# Main function
if __name__ == "__main__":
    s = Sim()
    s.run()
