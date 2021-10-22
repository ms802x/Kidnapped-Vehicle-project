#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {


// setting the number of particles 
num_particles = 100;
std::default_random_engine gen;
std::normal_distribution<double> x_pos(x, std[0]);
std::normal_distribution<double> y_pos(y, std[1]);
std::normal_distribution<double> angle(theta, std[2]);

for (int i = 0; i < num_particles; i++) {

Particle p;
 // set index to each particle 
 p.id = i;
 // specify its postion
 p.x = x_pos(gen);
 p.y = y_pos(gen);
 p.theta = angle(gen);
 //init the wieghts
 p.weight = 1.0;
 //push it in an array (array of instances)
 particles.push_back(p);
 }

 is_initialized = true;

}


void ParticleFilter::prediction(double delta_t, double std_pos[],
 double velocity, double yaw_rate) {
 /**
 * TODO: Add measurements to each particle and add random Gaussian noise.
 * NOTE: When adding noise you may find std::normal_distribution
 * and std::default_random_engine useful.
 * http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
 * http://www.cplusplus.com/reference/random/default_random_engine/
 */
 std::default_random_engine gen;
 std::normal_distribution<double> x_pos(0, std_pos[0]);
 std::normal_distribution<double> y_pos(0, std_pos[1]);
 std::normal_distribution<double> angle(0, std_pos[2]);
//putting int alone shows warnning, therefore, to avoid them I have used unsigned in 
 for (unsigned int i = 0; i < num_particles; i++) {
// this equation is only valid when yaw_rate !=0.
// getting excat zero depends on how many decimal place the data is, since our data can be up to 7 digits
 // I have added a this condition to cosider even the values close to zero
 if (fabs(yaw_rate) > 0.00001) {
   
 double y_t = yaw_rate * delta_t;
 particles[i].x += ( velocity / yaw_rate* (sin(particles[i].theta + y_t) - sin(particles[i].theta)));
 particles[i].y += ( velocity / yaw_rate* (cos(particles[i].theta) - cos(particles[i].theta + y_t)));
 particles[i].theta += y_t;
 
 }
   //this condition to account for 0 yaw rate. 
 else {

 particles[i].x += (velocity * delta_t * cos(particles[i].theta));
 particles[i].y += (velocity * delta_t* sin(particles[i].theta));
 }

 particles[i].x += x_pos(gen);
 particles[i].y += y_pos(gen);
 particles[i].theta += angle(gen);
 }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
 vector<LandmarkObs>& observations) {
 /**
 * TODO: Find the predicted measurement that is closest to each 
 * observed measurement and assign the observed measurement to this 
 * particular landmark.
 * NOTE: this method will NOT be called by the grading code. But you will 
 * probably find it useful to implement this method and use it as a helper 
 * during the updateWeights phase.
 */
 

for (unsigned int i = 0; i<observations.size(); i++){
// set arbitarry value 
 double min_value = 1000;
 double mark_id;
 
 for (unsigned int j = 0; j < predicted.size(); j++) {
 
 double val = dist(observations[i].x,observations[i].y, predicted[j].x, predicted[j].y);
 //store id corrsponding to the minmum value
 if (val < min_value){
 min_value = val;
 mark_id = predicted[j].id;
 }
 
 }
 observations[i].id = mark_id;}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
 const vector<LandmarkObs> &observations,
 const Map &map_landmarks) {
 /**
 * TODO: Update the weights of each particle using a mult-variate Gaussian
 * distribution. You can read more about this distribution here:
 * https://en.wikipedia.org/wiki/Multivariate_normal_distribution
 * NOTE: The observations are given in the VEHICLE'S coordinate system.
 * Your particles are located according to the MAP'S coordinate system.
 * You will need to transform between the two systems. Keep in mind that
 * this transformation requires both rotation AND translation (but no scaling).
 * The following is a good resource for the theory:
 * https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
 * and the following is a good resource for the actual equation to implement
 * (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
 */
 for (int i = 0; i < num_particles; i++)
 {
 // store postion and create vector for landmark
 double p_x = particles[i].x;
 double p_y = particles[i].y;
 double p_theta = particles[i].theta;
 vector<LandmarkObs> Mark_Inrange; ;

 for (unsigned int j=0;j<map_landmarks.landmark_list.size(); j++){
 // store landmark points
 double lx = map_landmarks.landmark_list[j].x_f;
 double ly = map_landmarks.landmark_list[j].y_f;
 double lid = map_landmarks.landmark_list[j].id_i;
 // to check if the points are in the sensor range 
if (fabs(lx - p_x) <= sensor_range && fabs(ly - p_y) <= sensor_range)
 Mark_Inrange.push_back(LandmarkObs{ lid, lx, ly }); 
 }
 

 vector<LandmarkObs> land_obs_pos;
 for (unsigned int j=0;j< observations.size(); j++){
//map these points
double otx= p_x + (cos(p_theta) * observations[j].x) - (sin(p_theta) * observations[j].y);
 double oty = p_y + (sin(p_theta) * observations[j].x) + (cos(p_theta) * observations[j].y);
 land_obs_pos.push_back(LandmarkObs{ observations[j].id, otx, oty });
 }
 

//auxilalry function for data assoctiontion
dataAssociation(Mark_Inrange, land_obs_pos);
// init partiles weights before gaussain
particles[i].weight = 1.0;

 for (unsigned int j = 0; j < land_obs_pos.size(); j++)
 {
 
 double pr_x,pr_y;
 double ob_x= land_obs_pos[j].x;
 double ob_y= land_obs_pos[j].y;
 int ob_id= land_obs_pos[j].id;
 

 for (unsigned int k = 0; k < Mark_Inrange.size(); k++) {
 int pr_id = Mark_Inrange[k].id;
 if(pr_id == ob_id ){
 pr_x = Mark_Inrange[k].x;
 pr_y = Mark_Inrange[k].y;
 }

 }

 double sigma_x = std_landmark[0];
 double sigma_y = std_landmark[1];
 double weight_v = ( 1 / (2 * M_PI * sigma_x * sigma_y)) * exp( -( pow(pr_x - ob_x, 2) / (2 * pow(sigma_x, 2)) + (pow(pr_y - ob_y, 2) / (2 * pow(sigma_y, 2))) ) );

particles[i].weight *= weight_v;
 }
 }

}

void ParticleFilter::resample() {

  
  // I tired to use the recomnded resambling function, however, it didn't work.


 std::default_random_engine gen;
 // vector to store weights
 vector<double> weights;
 // for storing max weight it is temp
 double max_Weight = 0;
 // iterate through the particles
 for (int i = 0; i < num_particles; i++) {
 weights.push_back(particles[i].weight);
 // store highest weight
 if (particles[i].weight > max_Weight) {
     max_Weight = particles[i].weight;
   
   
 }
 }

  
 std::uniform_real_distribution<double> dist_d(0.0, max_Weight);
 std::uniform_int_distribution<int> dist_i(0, num_particles - 1);
 vector<Particle> resampled;

  //Thi is the Resampling Wheel
 int index = dist_i(gen);
 double beta = 0.0;
 for (int i = 0; i < num_particles; i++) {
    beta = beta + dist_d(gen) * 2.0;
    while (beta > weights[index]) {
 
    beta -= weights[index];
    index = (index + 1) % num_particles;
 
 }
 
 resampled.push_back(particles[index]);
 }

 particles = resampled;

}


void ParticleFilter::SetAssociations(Particle& particle,
 const vector<int>& associations,
 const vector<double>& sense_x,
 const vector<double>& sense_y) {
 // particle: the particle to which assign each listed association,
 // and association's (x,y) world coordinates mapping
 // associations: The landmark id that goes along with each listed association
 // sense_x: the associations x mapping already converted to world coordinates
 // sense_y: the associations y mapping already converted to world coordinates
 particle.associations = associations;
 particle.sense_x = sense_x;
 particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
 vector<int> v = best.associations;
 std::stringstream ss;
 copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
 string s = ss.str();
 s = s.substr(0, s.length() - 1); // get rid of the trailing space
 return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
 vector<double> v;

 if (coord == "X") {
 v = best.sense_x;
 } else {
 v = best.sense_y;
 }

 std::stringstream ss;
 copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
 string s = ss.str();
 s = s.substr(0, s.length() - 1); // get rid of the trailing space
 return s;
}