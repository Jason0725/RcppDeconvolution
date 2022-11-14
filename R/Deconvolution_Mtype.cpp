#include <RcppArmadillo.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace std;
#include <omp.h>
// [[Rcpp::plugins(openmp)]]

arma::vec phiU(arma::vec t, double sigU){
  return (ones(t.n_elem)/(1.0+sigU*sigU*t%t));
}

arma::vec phiK(arma::vec t){
  
  return (1.0-(t % t)) % (1.0-(t % t)) % (1.0-(t % t));
}

//[[Rcpp::export]]
arma::vec EstimateCATE(arma::vec X_predict,arma::vec Y,arma::vec D,arma::vec Z, int errortype,double sigU,double h1,double h2,double rho1, double rho2){
  double dt = 0.001;
  vec t = linspace(-1,1,2001);
  int n = Z.n_elem;
  
  vec th = t/h1;
  
  int longt = 2001;
  int nx = X_predict.n_elem;
  
  mat OO = ones(n) * th.t();
  OO= Z * ones(1,2001) % OO;
  
  mat csO=cos(OO);
  mat snO=sin(OO);
  
  
  mat rehatphiW=sum(csO,0);
  mat imhatphiW=sum(snO,0);
  
  
  mat renum=D.t()*csO;
  mat imnum=D.t()*snO;
  
  
  mat xt= th* (ones(nx).t());
  xt=xt % (ones(longt) * X_predict.t());
  mat cxt=cos(xt);
  mat sxt=sin(xt);
  
  vec phiUth=phiU(th,sigU);
  
  vec matphiKU = phiK(t)/phiUth;
  
  mat matphiKUm = matphiKU.t();
  
  mat Den=(rehatphiW % matphiKUm)*cxt+(imhatphiW % matphiKUm)*sxt;
  mat Num=(renum % matphiKUm)*cxt+(imnum % matphiKUm)*sxt;
  
  uvec sss = find(Den<rho1);
  Den.elem(sss) = rho1 * ones(sss.n_elem);
  vec px = Num.t()/Den.t();
  
  th=t/h2;
  OO = ones(n) * th.t();
  OO= Z * ones(1,2001) % OO;
  
  csO=cos(OO);
  snO=sin(OO);
  rehatphiW=sum(csO,0);
  imhatphiW=sum(snO,0);
  
  renum=(Y%D).t()*csO;
  imnum=(Y%D).t()*snO;
  mat renum2=(Y%(1-D)).t()*csO;
  mat imnum2=(Y%(1-D)).t()*snO;
  
  xt= th* (ones(nx).t());
  xt=xt % (ones(longt) * X_predict.t());
  cxt=cos(xt);
  sxt=sin(xt);
  
  phiUth=phiU(th,sigU);
  matphiKU = phiK(t)/phiUth;
  matphiKUm = matphiKU.t();
  
  Den=(rehatphiW % matphiKUm)*cxt+(imhatphiW % matphiKUm)*sxt;
  Num=(renum % matphiKUm)*cxt+(imnum % matphiKUm)*sxt;
  
  mat Num2 = (renum2 % matphiKUm)*cxt+(imnum2 % matphiKUm)*sxt;
  
  sss = find(Den<rho2);
  Den.elem(sss) = rho2 * ones(sss.n_elem);
  
  return Num.t()/Den.t()/px - Num2.t()/Den.t()/(1-px);
}

//[[Rcpp::export]]
void test(){
  int i;
#pragma omp parallel for private(i)
  for(i=0;i<100;i++){
    EstimateCATE(randu(100),randu(500),randu(500),randu(500), 1,0.05,0.2,0.2,0.01,0.01);
  }
}
