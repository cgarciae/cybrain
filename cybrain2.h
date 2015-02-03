#ifndef __PYX_HAVE__cybrain2
#define __PYX_HAVE__cybrain2


#ifndef __PYX_HAVE_API__cybrain2

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(int) neuronCount;

#endif /* !__PYX_HAVE_API__cybrain2 */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initcybrain2(void);
#else
PyMODINIT_FUNC PyInit_cybrain2(void);
#endif

#endif /* !__PYX_HAVE__cybrain2 */
