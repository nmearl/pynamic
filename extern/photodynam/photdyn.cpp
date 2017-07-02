#include <boost/python.cpp>
#include "src/n_body_state.h"
#include "src/n_body_lc.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(photdyn)
{
    class_< NBodyState >("NBodyState", init<double, double, double, double, double, double, double, int, double>())
      .def("__call__", &NBodyState::operator())
      .def("get_bary_lt", &NBodyState::getBaryLT)
      .def("get_bary_lt", &NBodyState::V_Z_LT)
      .def("occult_n", &occultn);
}