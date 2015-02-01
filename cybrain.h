#ifndef __PYX_HAVE__cybrain
#define __PYX_HAVE__cybrain


#ifndef __PYX_HAVE_API__cybrain

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(int) neuronCount;

#endif /* !__PYX_HAVE_API__cybrain */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initcybrain(void);
#else
PyMODINIT_FUNC PyInit_cybrain(void);
#endif

#endif /* !__PYX_HAVE__cybrain */
