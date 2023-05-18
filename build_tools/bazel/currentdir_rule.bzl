def _current_dir_impl(ctx):
    """Dynamically determine the workspace root from the current context.

    The path is made available as a `WORKSPACE_ROOT` environmment variable and
    may for instance be consumed in the `toolchains` attributes for `cc_library`
    and `genrule` targets.
    """
    return [
        platform_common.TemplateVariableInfo({
            "CURRENT_DIR": "/".join([ctx.label.workspace_root,ctx.label.package])
        }),
    ]

current_dir = rule(
    implementation = _current_dir_impl,
    attrs = {},
)
