#include<iostream>
#include<cmath>
#include<limits>

using namespace std;

double funkcja(double x,double eps){
    double wynik=0.0;
    double poprzedniWynik=INFINITY;
    int n=0;
  while(abs(wynik-poprzedniWynik)>eps){
        double term=(pow(sin(n*pow(x,4)),2)*exp(-n)+cos(n*pow(x,4))*exp(-4*n));
        poprzedniWynik=wynik;
        wynik+=term;
       n++;
    }  
    return wynik;
}

int main(){
double eps=1e-10;
double x=1.0;
cout<<funkcja(x,eps)<<endl;
return 0;
}