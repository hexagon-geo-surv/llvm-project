configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SCHOOL_TEST_DEPENDS
  FileCheck count not
  school-opt
)

add_lit_testsuite(check-school "Running the school exercise tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SCHOOL_TEST_DEPENDS}
)
set_target_properties(check-school PROPERTIES FOLDER "Tests")
