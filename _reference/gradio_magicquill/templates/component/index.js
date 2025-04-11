const {
  SvelteComponent: d0,
  assign: p0,
  create_slot: g0,
  detach: m0,
  element: v0,
  get_all_dirty_from_scope: y0,
  get_slot_changes: b0,
  get_spread_update: x0,
  init: _0,
  insert: w0,
  safe_not_equal: C0,
  set_dynamic_element_data: Vc,
  set_style: Be,
  toggle_class: xn,
  transition_in: af,
  transition_out: of,
  update_slot_base: S0
} = window.__gradio__svelte__internal;
function E0(t) {
  let e, n, o;
  const d = (
    /*#slots*/
    t[18].default
  ), g = g0(
    d,
    t,
    /*$$scope*/
    t[17],
    null
  );
  let b = [
    {
      "data-testid": (
        /*test_id*/
        t[7]
      )
    },
    {
      id: (
        /*elem_id*/
        t[2]
      )
    },
    {
      class: n = "block " + /*elem_classes*/
        t[3].join(" ") + " svelte-nl1om8"
    }
  ], _ = {};
  for (let w = 0; w < b.length; w += 1)
    _ = p0(_, b[w]);
  return {
    c() {
      e = v0(
        /*tag*/
        t[14]
      ), g && g.c(), Vc(
        /*tag*/
        t[14]
      )(e, _), xn(
        e,
        "hidden",
        /*visible*/
        t[10] === !1
      ), xn(
        e,
        "padded",
        /*padding*/
        t[6]
      ), xn(
        e,
        "border_focus",
        /*border_mode*/
        t[5] === "focus"
      ), xn(
        e,
        "border_contrast",
        /*border_mode*/
        t[5] === "contrast"
      ), xn(e, "hide-container", !/*explicit_call*/
        t[8] && !/*container*/
        t[9]), Be(
          e,
          "height",
          /*get_dimension*/
          t[15](
            /*height*/
            t[0]
          )
        ), Be(e, "width", typeof /*width*/
          t[1] == "number" ? `calc(min(${/*width*/
        t[1]}px, 100%))` : (
          /*get_dimension*/
          t[15](
            /*width*/
            t[1]
          )
        )), Be(
          e,
          "border-style",
          /*variant*/
          t[4]
        ), Be(
          e,
          "overflow",
          /*allow_overflow*/
          t[11] ? "visible" : "hidden"
        ), Be(
          e,
          "flex-grow",
          /*scale*/
          t[12]
        ), Be(e, "min-width", `calc(min(${/*min_width*/
          t[13]}px, 100%))`), Be(e, "border-width", "var(--block-border-width)");
    },
    m(w, E) {
      w0(w, e, E), g && g.m(e, null), o = !0;
    },
    p(w, E) {
      g && g.p && (!o || E & /*$$scope*/
        131072) && S0(
          g,
          d,
          w,
          /*$$scope*/
          w[17],
          o ? b0(
            d,
            /*$$scope*/
            w[17],
            E,
            null
          ) : y0(
            /*$$scope*/
            w[17]
          ),
          null
        ), Vc(
          /*tag*/
          w[14]
        )(e, _ = x0(b, [
          (!o || E & /*test_id*/
            128) && {
              "data-testid": (
                /*test_id*/
                w[7]
              )
          },
          (!o || E & /*elem_id*/
            4) && {
              id: (
                /*elem_id*/
                w[2]
              )
          },
          (!o || E & /*elem_classes*/
            8 && n !== (n = "block " + /*elem_classes*/
              w[3].join(" ") + " svelte-nl1om8")) && { class: n }
        ])), xn(
          e,
          "hidden",
          /*visible*/
          w[10] === !1
        ), xn(
          e,
          "padded",
          /*padding*/
          w[6]
        ), xn(
          e,
          "border_focus",
          /*border_mode*/
          w[5] === "focus"
        ), xn(
          e,
          "border_contrast",
          /*border_mode*/
          w[5] === "contrast"
        ), xn(e, "hide-container", !/*explicit_call*/
          w[8] && !/*container*/
          w[9]), E & /*height*/
          1 && Be(
            e,
            "height",
            /*get_dimension*/
            w[15](
              /*height*/
              w[0]
            )
          ), E & /*width*/
          2 && Be(e, "width", typeof /*width*/
            w[1] == "number" ? `calc(min(${/*width*/
          w[1]}px, 100%))` : (
            /*get_dimension*/
            w[15](
              /*width*/
              w[1]
            )
          )), E & /*variant*/
          16 && Be(
            e,
            "border-style",
            /*variant*/
            w[4]
          ), E & /*allow_overflow*/
          2048 && Be(
            e,
            "overflow",
            /*allow_overflow*/
            w[11] ? "visible" : "hidden"
          ), E & /*scale*/
          4096 && Be(
            e,
            "flex-grow",
            /*scale*/
            w[12]
          ), E & /*min_width*/
          8192 && Be(e, "min-width", `calc(min(${/*min_width*/
            w[13]}px, 100%))`);
    },
    i(w) {
      o || (af(g, w), o = !0);
    },
    o(w) {
      of(g, w), o = !1;
    },
    d(w) {
      w && m0(e), g && g.d(w);
    }
  };
}
function k0(t) {
  let e, n = (
    /*tag*/
    t[14] && E0(t)
  );
  return {
    c() {
      n && n.c();
    },
    m(o, d) {
      n && n.m(o, d), e = !0;
    },
    p(o, [d]) {
      /*tag*/
      o[14] && n.p(o, d);
    },
    i(o) {
      e || (af(n, o), e = !0);
    },
    o(o) {
      of(n, o), e = !1;
    },
    d(o) {
      n && n.d(o);
    }
  };
}
function T0(t, e, n) {
  let { $$slots: o = {}, $$scope: d } = e, { height: g = void 0 } = e, { width: b = void 0 } = e, { elem_id: _ = "" } = e, { elem_classes: w = [] } = e, { variant: E = "solid" } = e, { border_mode: j = "base" } = e, { padding: H = !0 } = e, { type: G = "normal" } = e, { test_id: J = void 0 } = e, { explicit_call: nt = !1 } = e, { container: rt = !0 } = e, { visible: vt = !0 } = e, { allow_overflow: Y = !0 } = e, { scale: L = null } = e, { min_width: A = 0 } = e, tt = G === "fieldset" ? "fieldset" : "div";
  const dt = (it) => {
    if (it !== void 0) {
      if (typeof it == "number")
        return it + "px";
      if (typeof it == "string")
        return it;
    }
  };
  return t.$$set = (it) => {
    "height" in it && n(0, g = it.height), "width" in it && n(1, b = it.width), "elem_id" in it && n(2, _ = it.elem_id), "elem_classes" in it && n(3, w = it.elem_classes), "variant" in it && n(4, E = it.variant), "border_mode" in it && n(5, j = it.border_mode), "padding" in it && n(6, H = it.padding), "type" in it && n(16, G = it.type), "test_id" in it && n(7, J = it.test_id), "explicit_call" in it && n(8, nt = it.explicit_call), "container" in it && n(9, rt = it.container), "visible" in it && n(10, vt = it.visible), "allow_overflow" in it && n(11, Y = it.allow_overflow), "scale" in it && n(12, L = it.scale), "min_width" in it && n(13, A = it.min_width), "$$scope" in it && n(17, d = it.$$scope);
  }, [
      g,
      b,
      _,
      w,
      E,
      j,
      H,
      J,
      nt,
      rt,
      vt,
      Y,
      L,
      A,
      tt,
      dt,
      G,
      d,
      o
    ];
}
class O0 extends d0 {
  constructor(e) {
    super(), _0(this, e, T0, k0, C0, {
      height: 0,
      width: 1,
      elem_id: 2,
      elem_classes: 3,
      variant: 4,
      border_mode: 5,
      padding: 6,
      type: 16,
      test_id: 7,
      explicit_call: 8,
      container: 9,
      visible: 10,
      allow_overflow: 11,
      scale: 12,
      min_width: 13
    });
  }
}
const {
  SvelteComponent: P0,
  append: Us,
  attr: jn,
  bubble: M0,
  create_component: A0,
  destroy_component: D0,
  detach: sf,
  element: Hs,
  init: j0,
  insert: lf,
  listen: L0,
  mount_component: F0,
  safe_not_equal: I0,
  set_data: R0,
  set_style: Fi,
  space: B0,
  text: z0,
  toggle_class: Me,
  transition_in: N0,
  transition_out: U0
} = window.__gradio__svelte__internal;
function Gc(t) {
  let e, n;
  return {
    c() {
      e = Hs("span"), n = z0(
        /*label*/
        t[1]
      ), jn(e, "class", "svelte-1lrphxw");
    },
    m(o, d) {
      lf(o, e, d), Us(e, n);
    },
    p(o, d) {
      d & /*label*/
        2 && R0(
          n,
          /*label*/
          o[1]
        );
    },
    d(o) {
      o && sf(e);
    }
  };
}
function H0(t) {
  let e, n, o, d, g, b, _, w = (
    /*show_label*/
    t[2] && Gc(t)
  );
  return d = new /*Icon*/
    t[0]({}), {
    c() {
      e = Hs("button"), w && w.c(), n = B0(), o = Hs("div"), A0(d.$$.fragment), jn(o, "class", "svelte-1lrphxw"), Me(
        o,
        "small",
        /*size*/
        t[4] === "small"
      ), Me(
        o,
        "large",
        /*size*/
        t[4] === "large"
      ), Me(
        o,
        "medium",
        /*size*/
        t[4] === "medium"
      ), e.disabled = /*disabled*/
        t[7], jn(
          e,
          "aria-label",
          /*label*/
          t[1]
        ), jn(
          e,
          "aria-haspopup",
          /*hasPopup*/
          t[8]
        ), jn(
          e,
          "title",
          /*label*/
          t[1]
        ), jn(e, "class", "svelte-1lrphxw"), Me(
          e,
          "pending",
          /*pending*/
          t[3]
        ), Me(
          e,
          "padded",
          /*padded*/
          t[5]
        ), Me(
          e,
          "highlight",
          /*highlight*/
          t[6]
        ), Me(
          e,
          "transparent",
          /*transparent*/
          t[9]
        ), Fi(e, "color", !/*disabled*/
          t[7] && /*_color*/
          t[12] ? (
          /*_color*/
          t[12]
        ) : "var(--block-label-text-color)"), Fi(e, "--bg-color", /*disabled*/
          t[7] ? "auto" : (
            /*background*/
            t[10]
          )), Fi(
            e,
            "margin-left",
            /*offset*/
            t[11] + "px"
          );
    },
    m(E, j) {
      lf(E, e, j), w && w.m(e, null), Us(e, n), Us(e, o), F0(d, o, null), g = !0, b || (_ = L0(
        e,
        "click",
        /*click_handler*/
        t[14]
      ), b = !0);
    },
    p(E, [j]) {
      /*show_label*/
      E[2] ? w ? w.p(E, j) : (w = Gc(E), w.c(), w.m(e, n)) : w && (w.d(1), w = null), (!g || j & /*size*/
        16) && Me(
          o,
          "small",
          /*size*/
          E[4] === "small"
        ), (!g || j & /*size*/
          16) && Me(
            o,
            "large",
            /*size*/
            E[4] === "large"
          ), (!g || j & /*size*/
            16) && Me(
              o,
              "medium",
              /*size*/
              E[4] === "medium"
            ), (!g || j & /*disabled*/
              128) && (e.disabled = /*disabled*/
                E[7]), (!g || j & /*label*/
                  2) && jn(
                    e,
                    "aria-label",
                    /*label*/
                    E[1]
                  ), (!g || j & /*hasPopup*/
                    256) && jn(
                      e,
                      "aria-haspopup",
                      /*hasPopup*/
                      E[8]
                    ), (!g || j & /*label*/
                      2) && jn(
                        e,
                        "title",
                        /*label*/
                        E[1]
                      ), (!g || j & /*pending*/
                        8) && Me(
                          e,
                          "pending",
                          /*pending*/
                          E[3]
                        ), (!g || j & /*padded*/
                          32) && Me(
                            e,
                            "padded",
                            /*padded*/
                            E[5]
                          ), (!g || j & /*highlight*/
                            64) && Me(
                              e,
                              "highlight",
                              /*highlight*/
                              E[6]
                            ), (!g || j & /*transparent*/
                              512) && Me(
                                e,
                                "transparent",
                                /*transparent*/
                                E[9]
                              ), j & /*disabled, _color*/
                              4224 && Fi(e, "color", !/*disabled*/
                                E[7] && /*_color*/
                                E[12] ? (
                                /*_color*/
                                E[12]
                              ) : "var(--block-label-text-color)"), j & /*disabled, background*/
                              1152 && Fi(e, "--bg-color", /*disabled*/
                                E[7] ? "auto" : (
                                  /*background*/
                                  E[10]
                                )), j & /*offset*/
                                2048 && Fi(
                                  e,
                                  "margin-left",
                                  /*offset*/
                                  E[11] + "px"
                                );
    },
    i(E) {
      g || (N0(d.$$.fragment, E), g = !0);
    },
    o(E) {
      U0(d.$$.fragment, E), g = !1;
    },
    d(E) {
      E && sf(e), w && w.d(), D0(d), b = !1, _();
    }
  };
}
function W0(t, e, n) {
  let o, { Icon: d } = e, { label: g = "" } = e, { show_label: b = !1 } = e, { pending: _ = !1 } = e, { size: w = "small" } = e, { padded: E = !0 } = e, { highlight: j = !1 } = e, { disabled: H = !1 } = e, { hasPopup: G = !1 } = e, { color: J = "var(--block-label-text-color)" } = e, { transparent: nt = !1 } = e, { background: rt = "var(--background-fill-primary)" } = e, { offset: vt = 0 } = e;
  function Y(L) {
    M0.call(this, t, L);
  }
  return t.$$set = (L) => {
    "Icon" in L && n(0, d = L.Icon), "label" in L && n(1, g = L.label), "show_label" in L && n(2, b = L.show_label), "pending" in L && n(3, _ = L.pending), "size" in L && n(4, w = L.size), "padded" in L && n(5, E = L.padded), "highlight" in L && n(6, j = L.highlight), "disabled" in L && n(7, H = L.disabled), "hasPopup" in L && n(8, G = L.hasPopup), "color" in L && n(13, J = L.color), "transparent" in L && n(9, nt = L.transparent), "background" in L && n(10, rt = L.background), "offset" in L && n(11, vt = L.offset);
  }, t.$$.update = () => {
    t.$$.dirty & /*highlight, color*/
      8256 && n(12, o = j ? "var(--color-accent)" : J);
  }, [
      d,
      g,
      b,
      _,
      w,
      E,
      j,
      H,
      G,
      nt,
      rt,
      vt,
      o,
      J,
      Y
    ];
}
class X0 extends P0 {
  constructor(e) {
    super(), j0(this, e, W0, H0, I0, {
      Icon: 0,
      label: 1,
      show_label: 2,
      pending: 3,
      size: 4,
      padded: 5,
      highlight: 6,
      disabled: 7,
      hasPopup: 8,
      color: 13,
      transparent: 9,
      background: 10,
      offset: 11
    });
  }
}
const {
  SvelteComponent: Y0,
  append: is,
  attr: on,
  detach: V0,
  init: G0,
  insert: q0,
  noop: rs,
  safe_not_equal: Z0,
  set_style: _n,
  svg_element: xa
} = window.__gradio__svelte__internal;
function Q0(t) {
  let e, n, o, d;
  return {
    c() {
      e = xa("svg"), n = xa("g"), o = xa("path"), d = xa("path"), on(o, "d", "M18,6L6.087,17.913"), _n(o, "fill", "none"), _n(o, "fill-rule", "nonzero"), _n(o, "stroke-width", "2px"), on(n, "transform", "matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"), on(d, "d", "M4.364,4.364L19.636,19.636"), _n(d, "fill", "none"), _n(d, "fill-rule", "nonzero"), _n(d, "stroke-width", "2px"), on(e, "width", "100%"), on(e, "height", "100%"), on(e, "viewBox", "0 0 24 24"), on(e, "version", "1.1"), on(e, "xmlns", "http://www.w3.org/2000/svg"), on(e, "xmlns:xlink", "http://www.w3.org/1999/xlink"), on(e, "xml:space", "preserve"), on(e, "stroke", "currentColor"), _n(e, "fill-rule", "evenodd"), _n(e, "clip-rule", "evenodd"), _n(e, "stroke-linecap", "round"), _n(e, "stroke-linejoin", "round");
    },
    m(g, b) {
      q0(g, e, b), is(e, n), is(n, o), is(e, d);
    },
    p: rs,
    i: rs,
    o: rs,
    d(g) {
      g && V0(e);
    }
  };
}
class K0 extends Y0 {
  constructor(e) {
    super(), G0(this, e, null, Q0, Z0, {});
  }
}
const J0 = [
  { color: "red", primary: 600, secondary: 100 },
  { color: "green", primary: 600, secondary: 100 },
  { color: "blue", primary: 600, secondary: 100 },
  { color: "yellow", primary: 500, secondary: 100 },
  { color: "purple", primary: 600, secondary: 100 },
  { color: "teal", primary: 600, secondary: 100 },
  { color: "orange", primary: 600, secondary: 100 },
  { color: "cyan", primary: 600, secondary: 100 },
  { color: "lime", primary: 500, secondary: 100 },
  { color: "pink", primary: 600, secondary: 100 }
], qc = {
  inherit: "inherit",
  current: "currentColor",
  transparent: "transparent",
  black: "#000",
  white: "#fff",
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617"
  },
  gray: {
    50: "#f9fafb",
    100: "#f3f4f6",
    200: "#e5e7eb",
    300: "#d1d5db",
    400: "#9ca3af",
    500: "#6b7280",
    600: "#4b5563",
    700: "#374151",
    800: "#1f2937",
    900: "#111827",
    950: "#030712"
  },
  zinc: {
    50: "#fafafa",
    100: "#f4f4f5",
    200: "#e4e4e7",
    300: "#d4d4d8",
    400: "#a1a1aa",
    500: "#71717a",
    600: "#52525b",
    700: "#3f3f46",
    800: "#27272a",
    900: "#18181b",
    950: "#09090b"
  },
  neutral: {
    50: "#fafafa",
    100: "#f5f5f5",
    200: "#e5e5e5",
    300: "#d4d4d4",
    400: "#a3a3a3",
    500: "#737373",
    600: "#525252",
    700: "#404040",
    800: "#262626",
    900: "#171717",
    950: "#0a0a0a"
  },
  stone: {
    50: "#fafaf9",
    100: "#f5f5f4",
    200: "#e7e5e4",
    300: "#d6d3d1",
    400: "#a8a29e",
    500: "#78716c",
    600: "#57534e",
    700: "#44403c",
    800: "#292524",
    900: "#1c1917",
    950: "#0c0a09"
  },
  red: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
    950: "#450a0a"
  },
  orange: {
    50: "#fff7ed",
    100: "#ffedd5",
    200: "#fed7aa",
    300: "#fdba74",
    400: "#fb923c",
    500: "#f97316",
    600: "#ea580c",
    700: "#c2410c",
    800: "#9a3412",
    900: "#7c2d12",
    950: "#431407"
  },
  amber: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
    950: "#451a03"
  },
  yellow: {
    50: "#fefce8",
    100: "#fef9c3",
    200: "#fef08a",
    300: "#fde047",
    400: "#facc15",
    500: "#eab308",
    600: "#ca8a04",
    700: "#a16207",
    800: "#854d0e",
    900: "#713f12",
    950: "#422006"
  },
  lime: {
    50: "#f7fee7",
    100: "#ecfccb",
    200: "#d9f99d",
    300: "#bef264",
    400: "#a3e635",
    500: "#84cc16",
    600: "#65a30d",
    700: "#4d7c0f",
    800: "#3f6212",
    900: "#365314",
    950: "#1a2e05"
  },
  green: {
    50: "#f0fdf4",
    100: "#dcfce7",
    200: "#bbf7d0",
    300: "#86efac",
    400: "#4ade80",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
    800: "#166534",
    900: "#14532d",
    950: "#052e16"
  },
  emerald: {
    50: "#ecfdf5",
    100: "#d1fae5",
    200: "#a7f3d0",
    300: "#6ee7b7",
    400: "#34d399",
    500: "#10b981",
    600: "#059669",
    700: "#047857",
    800: "#065f46",
    900: "#064e3b",
    950: "#022c22"
  },
  teal: {
    50: "#f0fdfa",
    100: "#ccfbf1",
    200: "#99f6e4",
    300: "#5eead4",
    400: "#2dd4bf",
    500: "#14b8a6",
    600: "#0d9488",
    700: "#0f766e",
    800: "#115e59",
    900: "#134e4a",
    950: "#042f2e"
  },
  cyan: {
    50: "#ecfeff",
    100: "#cffafe",
    200: "#a5f3fc",
    300: "#67e8f9",
    400: "#22d3ee",
    500: "#06b6d4",
    600: "#0891b2",
    700: "#0e7490",
    800: "#155e75",
    900: "#164e63",
    950: "#083344"
  },
  sky: {
    50: "#f0f9ff",
    100: "#e0f2fe",
    200: "#bae6fd",
    300: "#7dd3fc",
    400: "#38bdf8",
    500: "#0ea5e9",
    600: "#0284c7",
    700: "#0369a1",
    800: "#075985",
    900: "#0c4a6e",
    950: "#082f49"
  },
  blue: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
    950: "#172554"
  },
  indigo: {
    50: "#eef2ff",
    100: "#e0e7ff",
    200: "#c7d2fe",
    300: "#a5b4fc",
    400: "#818cf8",
    500: "#6366f1",
    600: "#4f46e5",
    700: "#4338ca",
    800: "#3730a3",
    900: "#312e81",
    950: "#1e1b4b"
  },
  violet: {
    50: "#f5f3ff",
    100: "#ede9fe",
    200: "#ddd6fe",
    300: "#c4b5fd",
    400: "#a78bfa",
    500: "#8b5cf6",
    600: "#7c3aed",
    700: "#6d28d9",
    800: "#5b21b6",
    900: "#4c1d95",
    950: "#2e1065"
  },
  purple: {
    50: "#faf5ff",
    100: "#f3e8ff",
    200: "#e9d5ff",
    300: "#d8b4fe",
    400: "#c084fc",
    500: "#a855f7",
    600: "#9333ea",
    700: "#7e22ce",
    800: "#6b21a8",
    900: "#581c87",
    950: "#3b0764"
  },
  fuchsia: {
    50: "#fdf4ff",
    100: "#fae8ff",
    200: "#f5d0fe",
    300: "#f0abfc",
    400: "#e879f9",
    500: "#d946ef",
    600: "#c026d3",
    700: "#a21caf",
    800: "#86198f",
    900: "#701a75",
    950: "#4a044e"
  },
  pink: {
    50: "#fdf2f8",
    100: "#fce7f3",
    200: "#fbcfe8",
    300: "#f9a8d4",
    400: "#f472b6",
    500: "#ec4899",
    600: "#db2777",
    700: "#be185d",
    800: "#9d174d",
    900: "#831843",
    950: "#500724"
  },
  rose: {
    50: "#fff1f2",
    100: "#ffe4e6",
    200: "#fecdd3",
    300: "#fda4af",
    400: "#fb7185",
    500: "#f43f5e",
    600: "#e11d48",
    700: "#be123c",
    800: "#9f1239",
    900: "#881337",
    950: "#4c0519"
  }
};
J0.reduce(
  (t, { color: e, primary: n, secondary: o }) => ({
    ...t,
    [e]: {
      primary: qc[e][n],
      secondary: qc[e][o]
    }
  }),
  {}
);
function Bi(t) {
  let e = ["", "k", "M", "G", "T", "P", "E", "Z"], n = 0;
  for (; t > 1e3 && n < e.length - 1;)
    t /= 1e3, n++;
  let o = e[n];
  return (Number.isInteger(t) ? t : t.toFixed(1)) + o;
}
function Ga() {
}
function $0(t, e) {
  return t != t ? e == e : t !== e || t && typeof t == "object" || typeof t == "function";
}
const cf = typeof window < "u";
let Zc = cf ? () => window.performance.now() : () => Date.now(), uf = cf ? (t) => requestAnimationFrame(t) : Ga;
const Ki = /* @__PURE__ */ new Set();
function hf(t) {
  Ki.forEach((e) => {
    e.c(t) || (Ki.delete(e), e.f());
  }), Ki.size !== 0 && uf(hf);
}
function tg(t) {
  let e;
  return Ki.size === 0 && uf(hf), {
    promise: new Promise((n) => {
      Ki.add(e = { c: t, f: n });
    }),
    abort() {
      Ki.delete(e);
    }
  };
}
const Ii = [];
function eg(t, e = Ga) {
  let n;
  const o = /* @__PURE__ */ new Set();
  function d(_) {
    if ($0(t, _) && (t = _, n)) {
      const w = !Ii.length;
      for (const E of o)
        E[1](), Ii.push(E, t);
      if (w) {
        for (let E = 0; E < Ii.length; E += 2)
          Ii[E][0](Ii[E + 1]);
        Ii.length = 0;
      }
    }
  }
  function g(_) {
    d(_(t));
  }
  function b(_, w = Ga) {
    const E = [_, w];
    return o.add(E), o.size === 1 && (n = e(d, g) || Ga), _(t), () => {
      o.delete(E), o.size === 0 && n && (n(), n = null);
    };
  }
  return { set: d, update: g, subscribe: b };
}
function Qc(t) {
  return Object.prototype.toString.call(t) === "[object Date]";
}
function Ws(t, e, n, o) {
  if (typeof n == "number" || Qc(n)) {
    const d = o - n, g = (n - e) / (t.dt || 1 / 60), b = t.opts.stiffness * d, _ = t.opts.damping * g, w = (b - _) * t.inv_mass, E = (g + w) * t.dt;
    return Math.abs(E) < t.opts.precision && Math.abs(d) < t.opts.precision ? o : (t.settled = !1, Qc(n) ? new Date(n.getTime() + E) : n + E);
  } else {
    if (Array.isArray(n))
      return n.map(
        (d, g) => Ws(t, e[g], n[g], o[g])
      );
    if (typeof n == "object") {
      const d = {};
      for (const g in n)
        d[g] = Ws(t, e[g], n[g], o[g]);
      return d;
    } else
      throw new Error(`Cannot spring ${typeof n} values`);
  }
}
function Kc(t, e = {}) {
  const n = eg(t), { stiffness: o = 0.15, damping: d = 0.8, precision: g = 0.01 } = e;
  let b, _, w, E = t, j = t, H = 1, G = 0, J = !1;
  function nt(vt, Y = {}) {
    j = vt;
    const L = w = {};
    return t == null || Y.hard || rt.stiffness >= 1 && rt.damping >= 1 ? (J = !0, b = Zc(), E = vt, n.set(t = j), Promise.resolve()) : (Y.soft && (G = 1 / ((Y.soft === !0 ? 0.5 : +Y.soft) * 60), H = 0), _ || (b = Zc(), J = !1, _ = tg((A) => {
      if (J)
        return J = !1, _ = null, !1;
      H = Math.min(H + G, 1);
      const tt = {
        inv_mass: H,
        opts: rt,
        settled: !0,
        dt: (A - b) * 60 / 1e3
      }, dt = Ws(tt, E, t, j);
      return b = A, E = t, n.set(t = dt), tt.settled && (_ = null), !tt.settled;
    })), new Promise((A) => {
      _.promise.then(() => {
        L === w && A();
      });
    }));
  }
  const rt = {
    set: nt,
    update: (vt, Y) => nt(vt(j, t), Y),
    subscribe: n.subscribe,
    stiffness: o,
    damping: d,
    precision: g
  };
  return rt;
}
const {
  SvelteComponent: ng,
  append: sn,
  attr: Nt,
  component_subscribe: Jc,
  detach: ig,
  element: rg,
  init: ag,
  insert: og,
  noop: $c,
  safe_not_equal: sg,
  set_style: _a,
  svg_element: ln,
  toggle_class: tu
} = window.__gradio__svelte__internal, { onMount: lg } = window.__gradio__svelte__internal;
function cg(t) {
  let e, n, o, d, g, b, _, w, E, j, H, G;
  return {
    c() {
      e = rg("div"), n = ln("svg"), o = ln("g"), d = ln("path"), g = ln("path"), b = ln("path"), _ = ln("path"), w = ln("g"), E = ln("path"), j = ln("path"), H = ln("path"), G = ln("path"), Nt(d, "d", "M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z"), Nt(d, "fill", "#FF7C00"), Nt(d, "fill-opacity", "0.4"), Nt(d, "class", "svelte-43sxxs"), Nt(g, "d", "M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z"), Nt(g, "fill", "#FF7C00"), Nt(g, "class", "svelte-43sxxs"), Nt(b, "d", "M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z"), Nt(b, "fill", "#FF7C00"), Nt(b, "fill-opacity", "0.4"), Nt(b, "class", "svelte-43sxxs"), Nt(_, "d", "M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z"), Nt(_, "fill", "#FF7C00"), Nt(_, "class", "svelte-43sxxs"), _a(o, "transform", "translate(" + /*$top*/
        t[1][0] + "px, " + /*$top*/
        t[1][1] + "px)"), Nt(E, "d", "M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z"), Nt(E, "fill", "#FF7C00"), Nt(E, "fill-opacity", "0.4"), Nt(E, "class", "svelte-43sxxs"), Nt(j, "d", "M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z"), Nt(j, "fill", "#FF7C00"), Nt(j, "class", "svelte-43sxxs"), Nt(H, "d", "M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z"), Nt(H, "fill", "#FF7C00"), Nt(H, "fill-opacity", "0.4"), Nt(H, "class", "svelte-43sxxs"), Nt(G, "d", "M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z"), Nt(G, "fill", "#FF7C00"), Nt(G, "class", "svelte-43sxxs"), _a(w, "transform", "translate(" + /*$bottom*/
          t[2][0] + "px, " + /*$bottom*/
          t[2][1] + "px)"), Nt(n, "viewBox", "-1200 -1200 3000 3000"), Nt(n, "fill", "none"), Nt(n, "xmlns", "http://www.w3.org/2000/svg"), Nt(n, "class", "svelte-43sxxs"), Nt(e, "class", "svelte-43sxxs"), tu(
            e,
            "margin",
            /*margin*/
            t[0]
          );
    },
    m(J, nt) {
      og(J, e, nt), sn(e, n), sn(n, o), sn(o, d), sn(o, g), sn(o, b), sn(o, _), sn(n, w), sn(w, E), sn(w, j), sn(w, H), sn(w, G);
    },
    p(J, [nt]) {
      nt & /*$top*/
        2 && _a(o, "transform", "translate(" + /*$top*/
          J[1][0] + "px, " + /*$top*/
          J[1][1] + "px)"), nt & /*$bottom*/
          4 && _a(w, "transform", "translate(" + /*$bottom*/
            J[2][0] + "px, " + /*$bottom*/
            J[2][1] + "px)"), nt & /*margin*/
            1 && tu(
              e,
              "margin",
              /*margin*/
              J[0]
            );
    },
    i: $c,
    o: $c,
    d(J) {
      J && ig(e);
    }
  };
}
function ug(t, e, n) {
  let o, d;
  var g = this && this.__awaiter || function (J, nt, rt, vt) {
    function Y(L) {
      return L instanceof rt ? L : new rt(function (A) {
        A(L);
      });
    }
    return new (rt || (rt = Promise))(function (L, A) {
      function tt(ht) {
        try {
          it(vt.next(ht));
        } catch (v) {
          A(v);
        }
      }
      function dt(ht) {
        try {
          it(vt.throw(ht));
        } catch (v) {
          A(v);
        }
      }
      function it(ht) {
        ht.done ? L(ht.value) : Y(ht.value).then(tt, dt);
      }
      it((vt = vt.apply(J, nt || [])).next());
    });
  };
  let { margin: b = !0 } = e;
  const _ = Kc([0, 0]);
  Jc(t, _, (J) => n(1, o = J));
  const w = Kc([0, 0]);
  Jc(t, w, (J) => n(2, d = J));
  let E;
  function j() {
    return g(this, void 0, void 0, function* () {
      yield Promise.all([_.set([125, 140]), w.set([-125, -140])]), yield Promise.all([_.set([-125, 140]), w.set([125, -140])]), yield Promise.all([_.set([-125, 0]), w.set([125, -0])]), yield Promise.all([_.set([125, 0]), w.set([-125, 0])]);
    });
  }
  function H() {
    return g(this, void 0, void 0, function* () {
      yield j(), E || H();
    });
  }
  function G() {
    return g(this, void 0, void 0, function* () {
      yield Promise.all([_.set([125, 0]), w.set([-125, 0])]), H();
    });
  }
  return lg(() => (G(), () => E = !0)), t.$$set = (J) => {
    "margin" in J && n(0, b = J.margin);
  }, [b, o, d, _, w];
}
class hg extends ng {
  constructor(e) {
    super(), ag(this, e, ug, cg, sg, { margin: 0 });
  }
}
const {
  SvelteComponent: fg,
  append: xi,
  attr: gn,
  binding_callbacks: eu,
  check_outros: Xs,
  create_component: ff,
  create_slot: df,
  destroy_component: pf,
  destroy_each: gf,
  detach: At,
  element: On,
  empty: fr,
  ensure_array_like: so,
  get_all_dirty_from_scope: mf,
  get_slot_changes: vf,
  group_outros: Ys,
  init: dg,
  insert: Dt,
  mount_component: yf,
  noop: Vs,
  safe_not_equal: pg,
  set_data: en,
  set_style: Qn,
  space: Je,
  text: te,
  toggle_class: Ze,
  transition_in: pn,
  transition_out: Pn,
  update_slot_base: bf
} = window.__gradio__svelte__internal, { tick: gg } = window.__gradio__svelte__internal, { onDestroy: mg } = window.__gradio__svelte__internal, { createEventDispatcher: vg } = window.__gradio__svelte__internal, yg = (t) => ({}), nu = (t) => ({}), bg = (t) => ({}), iu = (t) => ({});
function ru(t, e, n) {
  const o = t.slice();
  return o[41] = e[n], o[43] = n, o;
}
function au(t, e, n) {
  const o = t.slice();
  return o[41] = e[n], o;
}
function xg(t) {
  let e, n, o, d, g = (
    /*i18n*/
    t[1]("common.error") + ""
  ), b, _, w;
  n = new X0({
    props: {
      Icon: K0,
      label: (
        /*i18n*/
        t[1]("common.clear")
      ),
      disabled: !1
    }
  }), n.$on(
    "click",
    /*click_handler*/
    t[32]
  );
  const E = (
    /*#slots*/
    t[30].error
  ), j = df(
    E,
    t,
    /*$$scope*/
    t[29],
    nu
  );
  return {
    c() {
      e = On("div"), ff(n.$$.fragment), o = Je(), d = On("span"), b = te(g), _ = Je(), j && j.c(), gn(e, "class", "clear-status svelte-v0wucf"), gn(d, "class", "error svelte-v0wucf");
    },
    m(H, G) {
      Dt(H, e, G), yf(n, e, null), Dt(H, o, G), Dt(H, d, G), xi(d, b), Dt(H, _, G), j && j.m(H, G), w = !0;
    },
    p(H, G) {
      const J = {};
      G[0] & /*i18n*/
        2 && (J.label = /*i18n*/
          H[1]("common.clear")), n.$set(J), (!w || G[0] & /*i18n*/
            2) && g !== (g = /*i18n*/
              H[1]("common.error") + "") && en(b, g), j && j.p && (!w || G[0] & /*$$scope*/
                536870912) && bf(
                  j,
                  E,
                  H,
                  /*$$scope*/
                  H[29],
                  w ? vf(
                    E,
                    /*$$scope*/
                    H[29],
                    G,
                    yg
                  ) : mf(
                    /*$$scope*/
                    H[29]
                  ),
                  nu
                );
    },
    i(H) {
      w || (pn(n.$$.fragment, H), pn(j, H), w = !0);
    },
    o(H) {
      Pn(n.$$.fragment, H), Pn(j, H), w = !1;
    },
    d(H) {
      H && (At(e), At(o), At(d), At(_)), pf(n), j && j.d(H);
    }
  };
}
function _g(t) {
  let e, n, o, d, g, b, _, w, E, j = (
    /*variant*/
    t[8] === "default" && /*show_eta_bar*/
    t[18] && /*show_progress*/
    t[6] === "full" && ou(t)
  );
  function H(A, tt) {
    if (
      /*progress*/
      A[7]
    ) return Sg;
    if (
      /*queue_position*/
      A[2] !== null && /*queue_size*/
      A[3] !== void 0 && /*queue_position*/
      A[2] >= 0
    ) return Cg;
    if (
      /*queue_position*/
      A[2] === 0
    ) return wg;
  }
  let G = H(t), J = G && G(t), nt = (
    /*timer*/
    t[5] && cu(t)
  );
  const rt = [Og, Tg], vt = [];
  function Y(A, tt) {
    return (
      /*last_progress_level*/
      A[15] != null ? 0 : (
        /*show_progress*/
        A[6] === "full" ? 1 : -1
      )
    );
  }
  ~(g = Y(t)) && (b = vt[g] = rt[g](t));
  let L = !/*timer*/
    t[5] && mu(t);
  return {
    c() {
      j && j.c(), e = Je(), n = On("div"), J && J.c(), o = Je(), nt && nt.c(), d = Je(), b && b.c(), _ = Je(), L && L.c(), w = fr(), gn(n, "class", "progress-text svelte-v0wucf"), Ze(
        n,
        "meta-text-center",
        /*variant*/
        t[8] === "center"
      ), Ze(
        n,
        "meta-text",
        /*variant*/
        t[8] === "default"
      );
    },
    m(A, tt) {
      j && j.m(A, tt), Dt(A, e, tt), Dt(A, n, tt), J && J.m(n, null), xi(n, o), nt && nt.m(n, null), Dt(A, d, tt), ~g && vt[g].m(A, tt), Dt(A, _, tt), L && L.m(A, tt), Dt(A, w, tt), E = !0;
    },
    p(A, tt) {
      /*variant*/
      A[8] === "default" && /*show_eta_bar*/
        A[18] && /*show_progress*/
        A[6] === "full" ? j ? j.p(A, tt) : (j = ou(A), j.c(), j.m(e.parentNode, e)) : j && (j.d(1), j = null), G === (G = H(A)) && J ? J.p(A, tt) : (J && J.d(1), J = G && G(A), J && (J.c(), J.m(n, o))), /*timer*/
        A[5] ? nt ? nt.p(A, tt) : (nt = cu(A), nt.c(), nt.m(n, null)) : nt && (nt.d(1), nt = null), (!E || tt[0] & /*variant*/
          256) && Ze(
            n,
            "meta-text-center",
            /*variant*/
            A[8] === "center"
          ), (!E || tt[0] & /*variant*/
            256) && Ze(
              n,
              "meta-text",
              /*variant*/
              A[8] === "default"
            );
      let dt = g;
      g = Y(A), g === dt ? ~g && vt[g].p(A, tt) : (b && (Ys(), Pn(vt[dt], 1, 1, () => {
        vt[dt] = null;
      }), Xs()), ~g ? (b = vt[g], b ? b.p(A, tt) : (b = vt[g] = rt[g](A), b.c()), pn(b, 1), b.m(_.parentNode, _)) : b = null), /*timer*/
        A[5] ? L && (Ys(), Pn(L, 1, 1, () => {
          L = null;
        }), Xs()) : L ? (L.p(A, tt), tt[0] & /*timer*/
          32 && pn(L, 1)) : (L = mu(A), L.c(), pn(L, 1), L.m(w.parentNode, w));
    },
    i(A) {
      E || (pn(b), pn(L), E = !0);
    },
    o(A) {
      Pn(b), Pn(L), E = !1;
    },
    d(A) {
      A && (At(e), At(n), At(d), At(_), At(w)), j && j.d(A), J && J.d(), nt && nt.d(), ~g && vt[g].d(A), L && L.d(A);
    }
  };
}
function ou(t) {
  let e, n = `translateX(${/*eta_level*/
    (t[17] || 0) * 100 - 100}%)`;
  return {
    c() {
      e = On("div"), gn(e, "class", "eta-bar svelte-v0wucf"), Qn(e, "transform", n);
    },
    m(o, d) {
      Dt(o, e, d);
    },
    p(o, d) {
      d[0] & /*eta_level*/
        131072 && n !== (n = `translateX(${/*eta_level*/
          (o[17] || 0) * 100 - 100}%)`) && Qn(e, "transform", n);
    },
    d(o) {
      o && At(e);
    }
  };
}
function wg(t) {
  let e;
  return {
    c() {
      e = te("processing |");
    },
    m(n, o) {
      Dt(n, e, o);
    },
    p: Vs,
    d(n) {
      n && At(e);
    }
  };
}
function Cg(t) {
  let e, n = (
    /*queue_position*/
    t[2] + 1 + ""
  ), o, d, g, b;
  return {
    c() {
      e = te("queue: "), o = te(n), d = te("/"), g = te(
        /*queue_size*/
        t[3]
      ), b = te(" |");
    },
    m(_, w) {
      Dt(_, e, w), Dt(_, o, w), Dt(_, d, w), Dt(_, g, w), Dt(_, b, w);
    },
    p(_, w) {
      w[0] & /*queue_position*/
        4 && n !== (n = /*queue_position*/
          _[2] + 1 + "") && en(o, n), w[0] & /*queue_size*/
          8 && en(
            g,
            /*queue_size*/
            _[3]
          );
    },
    d(_) {
      _ && (At(e), At(o), At(d), At(g), At(b));
    }
  };
}
function Sg(t) {
  let e, n = so(
    /*progress*/
    t[7]
  ), o = [];
  for (let d = 0; d < n.length; d += 1)
    o[d] = lu(au(t, n, d));
  return {
    c() {
      for (let d = 0; d < o.length; d += 1)
        o[d].c();
      e = fr();
    },
    m(d, g) {
      for (let b = 0; b < o.length; b += 1)
        o[b] && o[b].m(d, g);
      Dt(d, e, g);
    },
    p(d, g) {
      if (g[0] & /*progress*/
        128) {
        n = so(
          /*progress*/
          d[7]
        );
        let b;
        for (b = 0; b < n.length; b += 1) {
          const _ = au(d, n, b);
          o[b] ? o[b].p(_, g) : (o[b] = lu(_), o[b].c(), o[b].m(e.parentNode, e));
        }
        for (; b < o.length; b += 1)
          o[b].d(1);
        o.length = n.length;
      }
    },
    d(d) {
      d && At(e), gf(o, d);
    }
  };
}
function su(t) {
  let e, n = (
    /*p*/
    t[41].unit + ""
  ), o, d, g = " ", b;
  function _(j, H) {
    return (
      /*p*/
      j[41].length != null ? kg : Eg
    );
  }
  let w = _(t), E = w(t);
  return {
    c() {
      E.c(), e = Je(), o = te(n), d = te(" | "), b = te(g);
    },
    m(j, H) {
      E.m(j, H), Dt(j, e, H), Dt(j, o, H), Dt(j, d, H), Dt(j, b, H);
    },
    p(j, H) {
      w === (w = _(j)) && E ? E.p(j, H) : (E.d(1), E = w(j), E && (E.c(), E.m(e.parentNode, e))), H[0] & /*progress*/
        128 && n !== (n = /*p*/
          j[41].unit + "") && en(o, n);
    },
    d(j) {
      j && (At(e), At(o), At(d), At(b)), E.d(j);
    }
  };
}
function Eg(t) {
  let e = Bi(
    /*p*/
    t[41].index || 0
  ) + "", n;
  return {
    c() {
      n = te(e);
    },
    m(o, d) {
      Dt(o, n, d);
    },
    p(o, d) {
      d[0] & /*progress*/
        128 && e !== (e = Bi(
          /*p*/
          o[41].index || 0
        ) + "") && en(n, e);
    },
    d(o) {
      o && At(n);
    }
  };
}
function kg(t) {
  let e = Bi(
    /*p*/
    t[41].index || 0
  ) + "", n, o, d = Bi(
    /*p*/
    t[41].length
  ) + "", g;
  return {
    c() {
      n = te(e), o = te("/"), g = te(d);
    },
    m(b, _) {
      Dt(b, n, _), Dt(b, o, _), Dt(b, g, _);
    },
    p(b, _) {
      _[0] & /*progress*/
        128 && e !== (e = Bi(
          /*p*/
          b[41].index || 0
        ) + "") && en(n, e), _[0] & /*progress*/
        128 && d !== (d = Bi(
          /*p*/
          b[41].length
        ) + "") && en(g, d);
    },
    d(b) {
      b && (At(n), At(o), At(g));
    }
  };
}
function lu(t) {
  let e, n = (
    /*p*/
    t[41].index != null && su(t)
  );
  return {
    c() {
      n && n.c(), e = fr();
    },
    m(o, d) {
      n && n.m(o, d), Dt(o, e, d);
    },
    p(o, d) {
      /*p*/
      o[41].index != null ? n ? n.p(o, d) : (n = su(o), n.c(), n.m(e.parentNode, e)) : n && (n.d(1), n = null);
    },
    d(o) {
      o && At(e), n && n.d(o);
    }
  };
}
function cu(t) {
  let e, n = (
    /*eta*/
    t[0] ? `/${/*formatted_eta*/
      t[19]}` : ""
  ), o, d;
  return {
    c() {
      e = te(
        /*formatted_timer*/
        t[20]
      ), o = te(n), d = te("s");
    },
    m(g, b) {
      Dt(g, e, b), Dt(g, o, b), Dt(g, d, b);
    },
    p(g, b) {
      b[0] & /*formatted_timer*/
        1048576 && en(
          e,
          /*formatted_timer*/
          g[20]
        ), b[0] & /*eta, formatted_eta*/
        524289 && n !== (n = /*eta*/
          g[0] ? `/${/*formatted_eta*/
            g[19]}` : "") && en(o, n);
    },
    d(g) {
      g && (At(e), At(o), At(d));
    }
  };
}
function Tg(t) {
  let e, n;
  return e = new hg({
    props: {
      margin: (
        /*variant*/
        t[8] === "default"
      )
    }
  }), {
    c() {
      ff(e.$$.fragment);
    },
    m(o, d) {
      yf(e, o, d), n = !0;
    },
    p(o, d) {
      const g = {};
      d[0] & /*variant*/
        256 && (g.margin = /*variant*/
          o[8] === "default"), e.$set(g);
    },
    i(o) {
      n || (pn(e.$$.fragment, o), n = !0);
    },
    o(o) {
      Pn(e.$$.fragment, o), n = !1;
    },
    d(o) {
      pf(e, o);
    }
  };
}
function Og(t) {
  let e, n, o, d, g, b = `${/*last_progress_level*/
    t[15] * 100}%`, _ = (
      /*progress*/
      t[7] != null && uu(t)
    );
  return {
    c() {
      e = On("div"), n = On("div"), _ && _.c(), o = Je(), d = On("div"), g = On("div"), gn(n, "class", "progress-level-inner svelte-v0wucf"), gn(g, "class", "progress-bar svelte-v0wucf"), Qn(g, "width", b), gn(d, "class", "progress-bar-wrap svelte-v0wucf"), gn(e, "class", "progress-level svelte-v0wucf");
    },
    m(w, E) {
      Dt(w, e, E), xi(e, n), _ && _.m(n, null), xi(e, o), xi(e, d), xi(d, g), t[31](g);
    },
    p(w, E) {
      /*progress*/
      w[7] != null ? _ ? _.p(w, E) : (_ = uu(w), _.c(), _.m(n, null)) : _ && (_.d(1), _ = null), E[0] & /*last_progress_level*/
        32768 && b !== (b = `${/*last_progress_level*/
          w[15] * 100}%`) && Qn(g, "width", b);
    },
    i: Vs,
    o: Vs,
    d(w) {
      w && At(e), _ && _.d(), t[31](null);
    }
  };
}
function uu(t) {
  let e, n = so(
    /*progress*/
    t[7]
  ), o = [];
  for (let d = 0; d < n.length; d += 1)
    o[d] = gu(ru(t, n, d));
  return {
    c() {
      for (let d = 0; d < o.length; d += 1)
        o[d].c();
      e = fr();
    },
    m(d, g) {
      for (let b = 0; b < o.length; b += 1)
        o[b] && o[b].m(d, g);
      Dt(d, e, g);
    },
    p(d, g) {
      if (g[0] & /*progress_level, progress*/
        16512) {
        n = so(
          /*progress*/
          d[7]
        );
        let b;
        for (b = 0; b < n.length; b += 1) {
          const _ = ru(d, n, b);
          o[b] ? o[b].p(_, g) : (o[b] = gu(_), o[b].c(), o[b].m(e.parentNode, e));
        }
        for (; b < o.length; b += 1)
          o[b].d(1);
        o.length = n.length;
      }
    },
    d(d) {
      d && At(e), gf(o, d);
    }
  };
}
function hu(t) {
  let e, n, o, d, g = (
    /*i*/
    t[43] !== 0 && Pg()
  ), b = (
    /*p*/
    t[41].desc != null && fu(t)
  ), _ = (
    /*p*/
    t[41].desc != null && /*progress_level*/
    t[14] && /*progress_level*/
    t[14][
    /*i*/
    t[43]
    ] != null && du()
  ), w = (
    /*progress_level*/
    t[14] != null && pu(t)
  );
  return {
    c() {
      g && g.c(), e = Je(), b && b.c(), n = Je(), _ && _.c(), o = Je(), w && w.c(), d = fr();
    },
    m(E, j) {
      g && g.m(E, j), Dt(E, e, j), b && b.m(E, j), Dt(E, n, j), _ && _.m(E, j), Dt(E, o, j), w && w.m(E, j), Dt(E, d, j);
    },
    p(E, j) {
      /*p*/
      E[41].desc != null ? b ? b.p(E, j) : (b = fu(E), b.c(), b.m(n.parentNode, n)) : b && (b.d(1), b = null), /*p*/
        E[41].desc != null && /*progress_level*/
          E[14] && /*progress_level*/
          E[14][
          /*i*/
          E[43]
          ] != null ? _ || (_ = du(), _.c(), _.m(o.parentNode, o)) : _ && (_.d(1), _ = null), /*progress_level*/
        E[14] != null ? w ? w.p(E, j) : (w = pu(E), w.c(), w.m(d.parentNode, d)) : w && (w.d(1), w = null);
    },
    d(E) {
      E && (At(e), At(n), At(o), At(d)), g && g.d(E), b && b.d(E), _ && _.d(E), w && w.d(E);
    }
  };
}
function Pg(t) {
  let e;
  return {
    c() {
      e = te(" /");
    },
    m(n, o) {
      Dt(n, e, o);
    },
    d(n) {
      n && At(e);
    }
  };
}
function fu(t) {
  let e = (
    /*p*/
    t[41].desc + ""
  ), n;
  return {
    c() {
      n = te(e);
    },
    m(o, d) {
      Dt(o, n, d);
    },
    p(o, d) {
      d[0] & /*progress*/
        128 && e !== (e = /*p*/
          o[41].desc + "") && en(n, e);
    },
    d(o) {
      o && At(n);
    }
  };
}
function du(t) {
  let e;
  return {
    c() {
      e = te("-");
    },
    m(n, o) {
      Dt(n, e, o);
    },
    d(n) {
      n && At(e);
    }
  };
}
function pu(t) {
  let e = (100 * /*progress_level*/
    (t[14][
      /*i*/
      t[43]
    ] || 0)).toFixed(1) + "", n, o;
  return {
    c() {
      n = te(e), o = te("%");
    },
    m(d, g) {
      Dt(d, n, g), Dt(d, o, g);
    },
    p(d, g) {
      g[0] & /*progress_level*/
        16384 && e !== (e = (100 * /*progress_level*/
          (d[14][
            /*i*/
            d[43]
          ] || 0)).toFixed(1) + "") && en(n, e);
    },
    d(d) {
      d && (At(n), At(o));
    }
  };
}
function gu(t) {
  let e, n = (
    /*p*/
    (t[41].desc != null || /*progress_level*/
      t[14] && /*progress_level*/
      t[14][
      /*i*/
      t[43]
      ] != null) && hu(t)
  );
  return {
    c() {
      n && n.c(), e = fr();
    },
    m(o, d) {
      n && n.m(o, d), Dt(o, e, d);
    },
    p(o, d) {
      /*p*/
      o[41].desc != null || /*progress_level*/
        o[14] && /*progress_level*/
        o[14][
        /*i*/
        o[43]
        ] != null ? n ? n.p(o, d) : (n = hu(o), n.c(), n.m(e.parentNode, e)) : n && (n.d(1), n = null);
    },
    d(o) {
      o && At(e), n && n.d(o);
    }
  };
}
function mu(t) {
  let e, n, o, d;
  const g = (
    /*#slots*/
    t[30]["additional-loading-text"]
  ), b = df(
    g,
    t,
    /*$$scope*/
    t[29],
    iu
  );
  return {
    c() {
      e = On("p"), n = te(
        /*loading_text*/
        t[9]
      ), o = Je(), b && b.c(), gn(e, "class", "loading svelte-v0wucf");
    },
    m(_, w) {
      Dt(_, e, w), xi(e, n), Dt(_, o, w), b && b.m(_, w), d = !0;
    },
    p(_, w) {
      (!d || w[0] & /*loading_text*/
        512) && en(
          n,
          /*loading_text*/
          _[9]
        ), b && b.p && (!d || w[0] & /*$$scope*/
          536870912) && bf(
            b,
            g,
            _,
            /*$$scope*/
            _[29],
            d ? vf(
              g,
              /*$$scope*/
              _[29],
              w,
              bg
            ) : mf(
              /*$$scope*/
              _[29]
            ),
            iu
          );
    },
    i(_) {
      d || (pn(b, _), d = !0);
    },
    o(_) {
      Pn(b, _), d = !1;
    },
    d(_) {
      _ && (At(e), At(o)), b && b.d(_);
    }
  };
}
function Mg(t) {
  let e, n, o, d, g;
  const b = [_g, xg], _ = [];
  function w(E, j) {
    return (
      /*status*/
      E[4] === "pending" ? 0 : (
        /*status*/
        E[4] === "error" ? 1 : -1
      )
    );
  }
  return ~(n = w(t)) && (o = _[n] = b[n](t)), {
    c() {
      e = On("div"), o && o.c(), gn(e, "class", d = "wrap " + /*variant*/
        t[8] + " " + /*show_progress*/
        t[6] + " svelte-v0wucf"), Ze(e, "hide", !/*status*/
          t[4] || /*status*/
          t[4] === "complete" || /*show_progress*/
          t[6] === "hidden"), Ze(
            e,
            "translucent",
            /*variant*/
            t[8] === "center" && /*status*/
            (t[4] === "pending" || /*status*/
              t[4] === "error") || /*translucent*/
            t[11] || /*show_progress*/
            t[6] === "minimal"
          ), Ze(
            e,
            "generating",
            /*status*/
            t[4] === "generating" && /*show_progress*/
            t[6] === "full"
          ), Ze(
            e,
            "border",
            /*border*/
            t[12]
          ), Qn(
            e,
            "position",
            /*absolute*/
            t[10] ? "absolute" : "static"
          ), Qn(
            e,
            "padding",
            /*absolute*/
            t[10] ? "0" : "var(--size-8) 0"
          );
    },
    m(E, j) {
      Dt(E, e, j), ~n && _[n].m(e, null), t[33](e), g = !0;
    },
    p(E, j) {
      let H = n;
      n = w(E), n === H ? ~n && _[n].p(E, j) : (o && (Ys(), Pn(_[H], 1, 1, () => {
        _[H] = null;
      }), Xs()), ~n ? (o = _[n], o ? o.p(E, j) : (o = _[n] = b[n](E), o.c()), pn(o, 1), o.m(e, null)) : o = null), (!g || j[0] & /*variant, show_progress*/
        320 && d !== (d = "wrap " + /*variant*/
          E[8] + " " + /*show_progress*/
          E[6] + " svelte-v0wucf")) && gn(e, "class", d), (!g || j[0] & /*variant, show_progress, status, show_progress*/
            336) && Ze(e, "hide", !/*status*/
              E[4] || /*status*/
              E[4] === "complete" || /*show_progress*/
              E[6] === "hidden"), (!g || j[0] & /*variant, show_progress, variant, status, translucent, show_progress*/
                2384) && Ze(
                  e,
                  "translucent",
                  /*variant*/
                  E[8] === "center" && /*status*/
                  (E[4] === "pending" || /*status*/
                    E[4] === "error") || /*translucent*/
                  E[11] || /*show_progress*/
                  E[6] === "minimal"
                ), (!g || j[0] & /*variant, show_progress, status, show_progress*/
                  336) && Ze(
                    e,
                    "generating",
                    /*status*/
                    E[4] === "generating" && /*show_progress*/
                    E[6] === "full"
                  ), (!g || j[0] & /*variant, show_progress, border*/
                    4416) && Ze(
                      e,
                      "border",
                      /*border*/
                      E[12]
                    ), j[0] & /*absolute*/
                    1024 && Qn(
                      e,
                      "position",
                      /*absolute*/
                      E[10] ? "absolute" : "static"
                    ), j[0] & /*absolute*/
                    1024 && Qn(
                      e,
                      "padding",
                      /*absolute*/
                      E[10] ? "0" : "var(--size-8) 0"
                    );
    },
    i(E) {
      g || (pn(o), g = !0);
    },
    o(E) {
      Pn(o), g = !1;
    },
    d(E) {
      E && At(e), ~n && _[n].d(), t[33](null);
    }
  };
}
var Ag = function (t, e, n, o) {
  function d(g) {
    return g instanceof n ? g : new n(function (b) {
      b(g);
    });
  }
  return new (n || (n = Promise))(function (g, b) {
    function _(j) {
      try {
        E(o.next(j));
      } catch (H) {
        b(H);
      }
    }
    function w(j) {
      try {
        E(o.throw(j));
      } catch (H) {
        b(H);
      }
    }
    function E(j) {
      j.done ? g(j.value) : d(j.value).then(_, w);
    }
    E((o = o.apply(t, e || [])).next());
  });
};
let wa = [], as = !1;
function Dg(t) {
  return Ag(this, arguments, void 0, function* (e, n = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && n !== !0)) {
      if (wa.push(e), !as) as = !0;
      else return;
      yield gg(), requestAnimationFrame(() => {
        let o = [0, 0];
        for (let d = 0; d < wa.length; d++) {
          const b = wa[d].getBoundingClientRect();
          (d === 0 || b.top + window.scrollY <= o[0]) && (o[0] = b.top + window.scrollY, o[1] = d);
        }
        window.scrollTo({ top: o[0] - 20, behavior: "smooth" }), as = !1, wa = [];
      });
    }
  });
}
function jg(t, e, n) {
  let o, { $$slots: d = {}, $$scope: g } = e;
  this && this.__awaiter;
  const b = vg();
  let { i18n: _ } = e, { eta: w = null } = e, { queue_position: E } = e, { queue_size: j } = e, { status: H } = e, { scroll_to_output: G = !1 } = e, { timer: J = !0 } = e, { show_progress: nt = "full" } = e, { message: rt = null } = e, { progress: vt = null } = e, { variant: Y = "default" } = e, { loading_text: L = "Loading..." } = e, { absolute: A = !0 } = e, { translucent: tt = !1 } = e, { border: dt = !1 } = e, { autoscroll: it } = e, ht, v = !1, Ct = 0, F = 0, Et = null, Ht = null, It = 0, l = null, a, c = null, f = !0;
  const i = () => {
    n(0, w = n(27, Et = n(19, u = null))), n(25, Ct = performance.now()), n(26, F = 0), v = !0, r();
  };
  function r() {
    requestAnimationFrame(() => {
      n(26, F = (performance.now() - Ct) / 1e3), v && r();
    });
  }
  function s() {
    n(26, F = 0), n(0, w = n(27, Et = n(19, u = null))), v && (v = !1);
  }
  mg(() => {
    v && s();
  });
  let u = null;
  function h(y) {
    eu[y ? "unshift" : "push"](() => {
      c = y, n(16, c), n(7, vt), n(14, l), n(15, a);
    });
  }
  const p = () => {
    b("clear_status");
  };
  function m(y) {
    eu[y ? "unshift" : "push"](() => {
      ht = y, n(13, ht);
    });
  }
  return t.$$set = (y) => {
    "i18n" in y && n(1, _ = y.i18n), "eta" in y && n(0, w = y.eta), "queue_position" in y && n(2, E = y.queue_position), "queue_size" in y && n(3, j = y.queue_size), "status" in y && n(4, H = y.status), "scroll_to_output" in y && n(22, G = y.scroll_to_output), "timer" in y && n(5, J = y.timer), "show_progress" in y && n(6, nt = y.show_progress), "message" in y && n(23, rt = y.message), "progress" in y && n(7, vt = y.progress), "variant" in y && n(8, Y = y.variant), "loading_text" in y && n(9, L = y.loading_text), "absolute" in y && n(10, A = y.absolute), "translucent" in y && n(11, tt = y.translucent), "border" in y && n(12, dt = y.border), "autoscroll" in y && n(24, it = y.autoscroll), "$$scope" in y && n(29, g = y.$$scope);
  }, t.$$.update = () => {
    t.$$.dirty[0] & /*eta, old_eta, timer_start, eta_from_start*/
      436207617 && (w === null && n(0, w = Et), w != null && Et !== w && (n(28, Ht = (performance.now() - Ct) / 1e3 + w), n(19, u = Ht.toFixed(1)), n(27, Et = w))), t.$$.dirty[0] & /*eta_from_start, timer_diff*/
      335544320 && n(17, It = Ht === null || Ht <= 0 || !F ? null : Math.min(F / Ht, 1)), t.$$.dirty[0] & /*progress*/
      128 && vt != null && n(18, f = !1), t.$$.dirty[0] & /*progress, progress_level, progress_bar, last_progress_level*/
      114816 && (vt != null ? n(14, l = vt.map((y) => {
        if (y.index != null && y.length != null)
          return y.index / y.length;
        if (y.progress != null)
          return y.progress;
      })) : n(14, l = null), l ? (n(15, a = l[l.length - 1]), c && (a === 0 ? n(16, c.style.transition = "0", c) : n(16, c.style.transition = "150ms", c))) : n(15, a = void 0)), t.$$.dirty[0] & /*status*/
      16 && (H === "pending" ? i() : s()), t.$$.dirty[0] & /*el, scroll_to_output, status, autoscroll*/
      20979728 && ht && G && (H === "pending" || H === "complete") && Dg(ht, it), t.$$.dirty[0] & /*status, message*/
      8388624, t.$$.dirty[0] & /*timer_diff*/
      67108864 && n(20, o = F.toFixed(1));
  }, [
      w,
      _,
      E,
      j,
      H,
      J,
      nt,
      vt,
      Y,
      L,
      A,
      tt,
      dt,
      ht,
      l,
      a,
      c,
      It,
      f,
      u,
      o,
      b,
      G,
      rt,
      it,
      Ct,
      F,
      Et,
      Ht,
      g,
      d,
      h,
      p,
      m
    ];
}
class Lg extends fg {
  constructor(e) {
    super(), dg(
      this,
      e,
      jg,
      Mg,
      pg,
      {
        i18n: 1,
        eta: 0,
        queue_position: 2,
        queue_size: 3,
        status: 4,
        scroll_to_output: 22,
        timer: 5,
        show_progress: 6,
        message: 23,
        progress: 7,
        variant: 8,
        loading_text: 9,
        absolute: 10,
        translucent: 11,
        border: 12,
        autoscroll: 24
      },
      null,
      [-1, -1]
    );
  }
}
/*! @license DOMPurify 3.1.6 | (c) Cure53 and other contributors | Released under the Apache license 2.0 and Mozilla Public License 2.0 | github.com/cure53/DOMPurify/blob/3.1.6/LICENSE */
const {
  entries: xf,
  setPrototypeOf: vu,
  isFrozen: Fg,
  getPrototypeOf: Ig,
  getOwnPropertyDescriptor: Rg
} = Object;
let {
  freeze: ke,
  seal: nn,
  create: _f
} = Object, {
  apply: Gs,
  construct: qs
} = typeof Reflect < "u" && Reflect;
ke || (ke = function (e) {
  return e;
});
nn || (nn = function (e) {
  return e;
});
Gs || (Gs = function (e, n, o) {
  return e.apply(n, o);
});
qs || (qs = function (e, n) {
  return new e(...n);
});
const Ca = He(Array.prototype.forEach), yu = He(Array.prototype.pop), xr = He(Array.prototype.push), qa = He(String.prototype.toLowerCase), os = He(String.prototype.toString), bu = He(String.prototype.match), _r = He(String.prototype.replace), Bg = He(String.prototype.indexOf), zg = He(String.prototype.trim), un = He(Object.prototype.hasOwnProperty), Ce = He(RegExp.prototype.test), wr = Ng(TypeError);
function He(t) {
  return function (e) {
    for (var n = arguments.length, o = new Array(n > 1 ? n - 1 : 0), d = 1; d < n; d++)
      o[d - 1] = arguments[d];
    return Gs(t, e, o);
  };
}
function Ng(t) {
  return function () {
    for (var e = arguments.length, n = new Array(e), o = 0; o < e; o++)
      n[o] = arguments[o];
    return qs(t, n);
  };
}
function Lt(t, e) {
  let n = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : qa;
  vu && vu(t, null);
  let o = e.length;
  for (; o--;) {
    let d = e[o];
    if (typeof d == "string") {
      const g = n(d);
      g !== d && (Fg(e) || (e[o] = g), d = g);
    }
    t[d] = !0;
  }
  return t;
}
function Ug(t) {
  for (let e = 0; e < t.length; e++)
    un(t, e) || (t[e] = null);
  return t;
}
function mi(t) {
  const e = _f(null);
  for (const [n, o] of xf(t))
    un(t, n) && (Array.isArray(o) ? e[n] = Ug(o) : o && typeof o == "object" && o.constructor === Object ? e[n] = mi(o) : e[n] = o);
  return e;
}
function Cr(t, e) {
  for (; t !== null;) {
    const o = Rg(t, e);
    if (o) {
      if (o.get)
        return He(o.get);
      if (typeof o.value == "function")
        return He(o.value);
    }
    t = Ig(t);
  }
  function n() {
    return null;
  }
  return n;
}
const xu = ke(["a", "abbr", "acronym", "address", "area", "article", "aside", "audio", "b", "bdi", "bdo", "big", "blink", "blockquote", "body", "br", "button", "canvas", "caption", "center", "cite", "code", "col", "colgroup", "content", "data", "datalist", "dd", "decorator", "del", "details", "dfn", "dialog", "dir", "div", "dl", "dt", "element", "em", "fieldset", "figcaption", "figure", "font", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html", "i", "img", "input", "ins", "kbd", "label", "legend", "li", "main", "map", "mark", "marquee", "menu", "menuitem", "meter", "nav", "nobr", "ol", "optgroup", "option", "output", "p", "picture", "pre", "progress", "q", "rp", "rt", "ruby", "s", "samp", "section", "select", "shadow", "small", "source", "spacer", "span", "strike", "strong", "style", "sub", "summary", "sup", "table", "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "time", "tr", "track", "tt", "u", "ul", "var", "video", "wbr"]), ss = ke(["svg", "a", "altglyph", "altglyphdef", "altglyphitem", "animatecolor", "animatemotion", "animatetransform", "circle", "clippath", "defs", "desc", "ellipse", "filter", "font", "g", "glyph", "glyphref", "hkern", "image", "line", "lineargradient", "marker", "mask", "metadata", "mpath", "path", "pattern", "polygon", "polyline", "radialgradient", "rect", "stop", "style", "switch", "symbol", "text", "textpath", "title", "tref", "tspan", "view", "vkern"]), ls = ke(["feBlend", "feColorMatrix", "feComponentTransfer", "feComposite", "feConvolveMatrix", "feDiffuseLighting", "feDisplacementMap", "feDistantLight", "feDropShadow", "feFlood", "feFuncA", "feFuncB", "feFuncG", "feFuncR", "feGaussianBlur", "feImage", "feMerge", "feMergeNode", "feMorphology", "feOffset", "fePointLight", "feSpecularLighting", "feSpotLight", "feTile", "feTurbulence"]), Hg = ke(["animate", "color-profile", "cursor", "discard", "font-face", "font-face-format", "font-face-name", "font-face-src", "font-face-uri", "foreignobject", "hatch", "hatchpath", "mesh", "meshgradient", "meshpatch", "meshrow", "missing-glyph", "script", "set", "solidcolor", "unknown", "use"]), cs = ke(["math", "menclose", "merror", "mfenced", "mfrac", "mglyph", "mi", "mlabeledtr", "mmultiscripts", "mn", "mo", "mover", "mpadded", "mphantom", "mroot", "mrow", "ms", "mspace", "msqrt", "mstyle", "msub", "msup", "msubsup", "mtable", "mtd", "mtext", "mtr", "munder", "munderover", "mprescripts"]), Wg = ke(["maction", "maligngroup", "malignmark", "mlongdiv", "mscarries", "mscarry", "msgroup", "mstack", "msline", "msrow", "semantics", "annotation", "annotation-xml", "mprescripts", "none"]), _u = ke(["#text"]), wu = ke(["accept", "action", "align", "alt", "autocapitalize", "autocomplete", "autopictureinpicture", "autoplay", "background", "bgcolor", "border", "capture", "cellpadding", "cellspacing", "checked", "cite", "class", "clear", "color", "cols", "colspan", "controls", "controlslist", "coords", "crossorigin", "datetime", "decoding", "default", "dir", "disabled", "disablepictureinpicture", "disableremoteplayback", "download", "draggable", "enctype", "enterkeyhint", "face", "for", "headers", "height", "hidden", "high", "href", "hreflang", "id", "inputmode", "integrity", "ismap", "kind", "label", "lang", "list", "loading", "loop", "low", "max", "maxlength", "media", "method", "min", "minlength", "multiple", "muted", "name", "nonce", "noshade", "novalidate", "nowrap", "open", "optimum", "pattern", "placeholder", "playsinline", "popover", "popovertarget", "popovertargetaction", "poster", "preload", "pubdate", "radiogroup", "readonly", "rel", "required", "rev", "reversed", "role", "rows", "rowspan", "spellcheck", "scope", "selected", "shape", "size", "sizes", "span", "srclang", "start", "src", "srcset", "step", "style", "summary", "tabindex", "title", "translate", "type", "usemap", "valign", "value", "width", "wrap", "xmlns", "slot"]), us = ke(["accent-height", "accumulate", "additive", "alignment-baseline", "ascent", "attributename", "attributetype", "azimuth", "basefrequency", "baseline-shift", "begin", "bias", "by", "class", "clip", "clippathunits", "clip-path", "clip-rule", "color", "color-interpolation", "color-interpolation-filters", "color-profile", "color-rendering", "cx", "cy", "d", "dx", "dy", "diffuseconstant", "direction", "display", "divisor", "dur", "edgemode", "elevation", "end", "fill", "fill-opacity", "fill-rule", "filter", "filterunits", "flood-color", "flood-opacity", "font-family", "font-size", "font-size-adjust", "font-stretch", "font-style", "font-variant", "font-weight", "fx", "fy", "g1", "g2", "glyph-name", "glyphref", "gradientunits", "gradienttransform", "height", "href", "id", "image-rendering", "in", "in2", "k", "k1", "k2", "k3", "k4", "kerning", "keypoints", "keysplines", "keytimes", "lang", "lengthadjust", "letter-spacing", "kernelmatrix", "kernelunitlength", "lighting-color", "local", "marker-end", "marker-mid", "marker-start", "markerheight", "markerunits", "markerwidth", "maskcontentunits", "maskunits", "max", "mask", "media", "method", "mode", "min", "name", "numoctaves", "offset", "operator", "opacity", "order", "orient", "orientation", "origin", "overflow", "paint-order", "path", "pathlength", "patterncontentunits", "patterntransform", "patternunits", "points", "preservealpha", "preserveaspectratio", "primitiveunits", "r", "rx", "ry", "radius", "refx", "refy", "repeatcount", "repeatdur", "restart", "result", "rotate", "scale", "seed", "shape-rendering", "specularconstant", "specularexponent", "spreadmethod", "startoffset", "stddeviation", "stitchtiles", "stop-color", "stop-opacity", "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "stroke-opacity", "stroke", "stroke-width", "style", "surfacescale", "systemlanguage", "tabindex", "targetx", "targety", "transform", "transform-origin", "text-anchor", "text-decoration", "text-rendering", "textlength", "type", "u1", "u2", "unicode", "values", "viewbox", "visibility", "version", "vert-adv-y", "vert-origin-x", "vert-origin-y", "width", "word-spacing", "wrap", "writing-mode", "xchannelselector", "ychannelselector", "x", "x1", "x2", "xmlns", "y", "y1", "y2", "z", "zoomandpan"]), Cu = ke(["accent", "accentunder", "align", "bevelled", "close", "columnsalign", "columnlines", "columnspan", "denomalign", "depth", "dir", "display", "displaystyle", "encoding", "fence", "frame", "height", "href", "id", "largeop", "length", "linethickness", "lspace", "lquote", "mathbackground", "mathcolor", "mathsize", "mathvariant", "maxsize", "minsize", "movablelimits", "notation", "numalign", "open", "rowalign", "rowlines", "rowspacing", "rowspan", "rspace", "rquote", "scriptlevel", "scriptminsize", "scriptsizemultiplier", "selection", "separator", "separators", "stretchy", "subscriptshift", "supscriptshift", "symmetric", "voffset", "width", "xmlns"]), Sa = ke(["xlink:href", "xml:id", "xlink:title", "xml:space", "xmlns:xlink"]), Xg = nn(/\{\{[\w\W]*|[\w\W]*\}\}/gm), Yg = nn(/<%[\w\W]*|[\w\W]*%>/gm), Vg = nn(/\${[\w\W]*}/gm), Gg = nn(/^data-[\-\w.\u00B7-\uFFFF]/), qg = nn(/^aria-[\-\w]+$/), wf = nn(
  /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|sms|cid|xmpp):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i
  // eslint-disable-line no-useless-escape
), Zg = nn(/^(?:\w+script|data):/i), Qg = nn(
  /[\u0000-\u0020\u00A0\u1680\u180E\u2000-\u2029\u205F\u3000]/g
  // eslint-disable-line no-control-regex
), Cf = nn(/^html$/i), Kg = nn(/^[a-z][.\w]*(-[.\w]+)+$/i);
var Su = /* @__PURE__ */ Object.freeze({
  __proto__: null,
  MUSTACHE_EXPR: Xg,
  ERB_EXPR: Yg,
  TMPLIT_EXPR: Vg,
  DATA_ATTR: Gg,
  ARIA_ATTR: qg,
  IS_ALLOWED_URI: wf,
  IS_SCRIPT_OR_DATA: Zg,
  ATTR_WHITESPACE: Qg,
  DOCTYPE_NAME: Cf,
  CUSTOM_ELEMENT: Kg
});
const Sr = {
  element: 1,
  attribute: 2,
  text: 3,
  cdataSection: 4,
  entityReference: 5,
  // Deprecated
  entityNode: 6,
  // Deprecated
  progressingInstruction: 7,
  comment: 8,
  document: 9,
  documentType: 10,
  documentFragment: 11,
  notation: 12
  // Deprecated
}, Jg = function () {
  return typeof window > "u" ? null : window;
}, $g = function (e, n) {
  if (typeof e != "object" || typeof e.createPolicy != "function")
    return null;
  let o = null;
  const d = "data-tt-policy-suffix";
  n && n.hasAttribute(d) && (o = n.getAttribute(d));
  const g = "dompurify" + (o ? "#" + o : "");
  try {
    return e.createPolicy(g, {
      createHTML(b) {
        return b;
      },
      createScriptURL(b) {
        return b;
      }
    });
  } catch {
    return console.warn("TrustedTypes policy " + g + " could not be created."), null;
  }
};
function Sf() {
  let t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : Jg();
  const e = (wt) => Sf(wt);
  if (e.version = "3.1.6", e.removed = [], !t || !t.document || t.document.nodeType !== Sr.document)
    return e.isSupported = !1, e;
  let {
    document: n
  } = t;
  const o = n, d = o.currentScript, {
    DocumentFragment: g,
    HTMLTemplateElement: b,
    Node: _,
    Element: w,
    NodeFilter: E,
    NamedNodeMap: j = t.NamedNodeMap || t.MozNamedAttrMap,
    HTMLFormElement: H,
    DOMParser: G,
    trustedTypes: J
  } = t, nt = w.prototype, rt = Cr(nt, "cloneNode"), vt = Cr(nt, "remove"), Y = Cr(nt, "nextSibling"), L = Cr(nt, "childNodes"), A = Cr(nt, "parentNode");
  if (typeof b == "function") {
    const wt = n.createElement("template");
    wt.content && wt.content.ownerDocument && (n = wt.content.ownerDocument);
  }
  let tt, dt = "";
  const {
    implementation: it,
    createNodeIterator: ht,
    createDocumentFragment: v,
    getElementsByTagName: Ct
  } = n, {
    importNode: F
  } = o;
  let Et = {};
  e.isSupported = typeof xf == "function" && typeof A == "function" && it && it.createHTMLDocument !== void 0;
  const {
    MUSTACHE_EXPR: Ht,
    ERB_EXPR: It,
    TMPLIT_EXPR: l,
    DATA_ATTR: a,
    ARIA_ATTR: c,
    IS_SCRIPT_OR_DATA: f,
    ATTR_WHITESPACE: i,
    CUSTOM_ELEMENT: r
  } = Su;
  let {
    IS_ALLOWED_URI: s
  } = Su, u = null;
  const h = Lt({}, [...xu, ...ss, ...ls, ...cs, ..._u]);
  let p = null;
  const m = Lt({}, [...wu, ...us, ...Cu, ...Sa]);
  let y = Object.seal(_f(null, {
    tagNameCheck: {
      writable: !0,
      configurable: !1,
      enumerable: !0,
      value: null
    },
    attributeNameCheck: {
      writable: !0,
      configurable: !1,
      enumerable: !0,
      value: null
    },
    allowCustomizedBuiltInElements: {
      writable: !0,
      configurable: !1,
      enumerable: !0,
      value: !1
    }
  })), x = null, S = null, k = !0, D = !0, M = !1, C = !0, O = !1, N = !0, W = !1, X = !1, I = !1, T = !1, R = !1, Q = !1, K = !0, U = !1;
  const B = "user-content-";
  let Z = !0, z = !1, V = {}, et = null;
  const lt = Lt({}, ["annotation-xml", "audio", "colgroup", "desc", "foreignobject", "head", "iframe", "math", "mi", "mn", "mo", "ms", "mtext", "noembed", "noframes", "noscript", "plaintext", "script", "style", "svg", "template", "thead", "title", "video", "xmp"]);
  let at = null;
  const ft = Lt({}, ["audio", "video", "img", "source", "image", "track"]);
  let ot = null;
  const ct = Lt({}, ["alt", "class", "for", "id", "label", "name", "pattern", "placeholder", "role", "summary", "title", "value", "style", "xmlns"]), st = "http://www.w3.org/1998/Math/MathML", yt = "http://www.w3.org/2000/svg", gt = "http://www.w3.org/1999/xhtml";
  let Tt = gt, kt = !1, jt = null;
  const Kt = Lt({}, [st, yt, gt], os);
  let Rt = null;
  const Bt = ["application/xhtml+xml", "text/html"], se = "text/html";
  let zt = null, ie = null;
  const fi = n.createElement("form"), mr = function (q) {
    return q instanceof RegExp || q instanceof Function;
  }, vr = function () {
    let q = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
    if (!(ie && ie === q)) {
      if ((!q || typeof q != "object") && (q = {}), q = mi(q), Rt = // eslint-disable-next-line unicorn/prefer-includes
        Bt.indexOf(q.PARSER_MEDIA_TYPE) === -1 ? se : q.PARSER_MEDIA_TYPE, zt = Rt === "application/xhtml+xml" ? os : qa, u = un(q, "ALLOWED_TAGS") ? Lt({}, q.ALLOWED_TAGS, zt) : h, p = un(q, "ALLOWED_ATTR") ? Lt({}, q.ALLOWED_ATTR, zt) : m, jt = un(q, "ALLOWED_NAMESPACES") ? Lt({}, q.ALLOWED_NAMESPACES, os) : Kt, ot = un(q, "ADD_URI_SAFE_ATTR") ? Lt(
          mi(ct),
          // eslint-disable-line indent
          q.ADD_URI_SAFE_ATTR,
          // eslint-disable-line indent
          zt
          // eslint-disable-line indent
        ) : ct, at = un(q, "ADD_DATA_URI_TAGS") ? Lt(
          mi(ft),
          // eslint-disable-line indent
          q.ADD_DATA_URI_TAGS,
          // eslint-disable-line indent
          zt
          // eslint-disable-line indent
        ) : ft, et = un(q, "FORBID_CONTENTS") ? Lt({}, q.FORBID_CONTENTS, zt) : lt, x = un(q, "FORBID_TAGS") ? Lt({}, q.FORBID_TAGS, zt) : {}, S = un(q, "FORBID_ATTR") ? Lt({}, q.FORBID_ATTR, zt) : {}, V = un(q, "USE_PROFILES") ? q.USE_PROFILES : !1, k = q.ALLOW_ARIA_ATTR !== !1, D = q.ALLOW_DATA_ATTR !== !1, M = q.ALLOW_UNKNOWN_PROTOCOLS || !1, C = q.ALLOW_SELF_CLOSE_IN_ATTR !== !1, O = q.SAFE_FOR_TEMPLATES || !1, N = q.SAFE_FOR_XML !== !1, W = q.WHOLE_DOCUMENT || !1, T = q.RETURN_DOM || !1, R = q.RETURN_DOM_FRAGMENT || !1, Q = q.RETURN_TRUSTED_TYPE || !1, I = q.FORCE_BODY || !1, K = q.SANITIZE_DOM !== !1, U = q.SANITIZE_NAMED_PROPS || !1, Z = q.KEEP_CONTENT !== !1, z = q.IN_PLACE || !1, s = q.ALLOWED_URI_REGEXP || wf, Tt = q.NAMESPACE || gt, y = q.CUSTOM_ELEMENT_HANDLING || {}, q.CUSTOM_ELEMENT_HANDLING && mr(q.CUSTOM_ELEMENT_HANDLING.tagNameCheck) && (y.tagNameCheck = q.CUSTOM_ELEMENT_HANDLING.tagNameCheck), q.CUSTOM_ELEMENT_HANDLING && mr(q.CUSTOM_ELEMENT_HANDLING.attributeNameCheck) && (y.attributeNameCheck = q.CUSTOM_ELEMENT_HANDLING.attributeNameCheck), q.CUSTOM_ELEMENT_HANDLING && typeof q.CUSTOM_ELEMENT_HANDLING.allowCustomizedBuiltInElements == "boolean" && (y.allowCustomizedBuiltInElements = q.CUSTOM_ELEMENT_HANDLING.allowCustomizedBuiltInElements), O && (D = !1), R && (T = !0), V && (u = Lt({}, _u), p = [], V.html === !0 && (Lt(u, xu), Lt(p, wu)), V.svg === !0 && (Lt(u, ss), Lt(p, us), Lt(p, Sa)), V.svgFilters === !0 && (Lt(u, ls), Lt(p, us), Lt(p, Sa)), V.mathMl === !0 && (Lt(u, cs), Lt(p, Cu), Lt(p, Sa))), q.ADD_TAGS && (u === h && (u = mi(u)), Lt(u, q.ADD_TAGS, zt)), q.ADD_ATTR && (p === m && (p = mi(p)), Lt(p, q.ADD_ATTR, zt)), q.ADD_URI_SAFE_ATTR && Lt(ot, q.ADD_URI_SAFE_ATTR, zt), q.FORBID_CONTENTS && (et === lt && (et = mi(et)), Lt(et, q.FORBID_CONTENTS, zt)), Z && (u["#text"] = !0), W && Lt(u, ["html", "head", "body"]), u.table && (Lt(u, ["tbody"]), delete x.tbody), q.TRUSTED_TYPES_POLICY) {
        if (typeof q.TRUSTED_TYPES_POLICY.createHTML != "function")
          throw wr('TRUSTED_TYPES_POLICY configuration option must provide a "createHTML" hook.');
        if (typeof q.TRUSTED_TYPES_POLICY.createScriptURL != "function")
          throw wr('TRUSTED_TYPES_POLICY configuration option must provide a "createScriptURL" hook.');
        tt = q.TRUSTED_TYPES_POLICY, dt = tt.createHTML("");
      } else
        tt === void 0 && (tt = $g(J, d)), tt !== null && typeof dt == "string" && (dt = tt.createHTML(""));
      ke && ke(q), ie = q;
    }
  }, ma = Lt({}, ["mi", "mo", "mn", "ms", "mtext"]), di = Lt({}, ["foreignobject", "annotation-xml"]), pi = Lt({}, ["title", "style", "font", "a", "script"]), Li = Lt({}, [...ss, ...ls, ...Hg]), va = Lt({}, [...cs, ...Wg]), ns = function (q) {
    let mt = A(q);
    (!mt || !mt.tagName) && (mt = {
      namespaceURI: Tt,
      tagName: "template"
    });
    const _t = qa(q.tagName), Wt = qa(mt.tagName);
    return jt[q.namespaceURI] ? q.namespaceURI === yt ? mt.namespaceURI === gt ? _t === "svg" : mt.namespaceURI === st ? _t === "svg" && (Wt === "annotation-xml" || ma[Wt]) : !!Li[_t] : q.namespaceURI === st ? mt.namespaceURI === gt ? _t === "math" : mt.namespaceURI === yt ? _t === "math" && di[Wt] : !!va[_t] : q.namespaceURI === gt ? mt.namespaceURI === yt && !di[Wt] || mt.namespaceURI === st && !ma[Wt] ? !1 : !va[_t] && (pi[_t] || !Li[_t]) : !!(Rt === "application/xhtml+xml" && jt[q.namespaceURI]) : !1;
  }, Ve = function (q) {
    xr(e.removed, {
      element: q
    });
    try {
      A(q).removeChild(q);
    } catch {
      vt(q);
    }
  }, Ge = function (q, mt) {
    try {
      xr(e.removed, {
        attribute: mt.getAttributeNode(q),
        from: mt
      });
    } catch {
      xr(e.removed, {
        attribute: null,
        from: mt
      });
    }
    if (mt.removeAttribute(q), q === "is" && !p[q])
      if (T || R)
        try {
          Ve(mt);
        } catch {
        }
      else
        try {
          mt.setAttribute(q, "");
        } catch {
        }
  }, ya = function (q) {
    let mt = null, _t = null;
    if (I)
      q = "<remove></remove>" + q;
    else {
      const de = bu(q, /^[\r\n\t ]+/);
      _t = de && de[0];
    }
    Rt === "application/xhtml+xml" && Tt === gt && (q = '<html xmlns="http://www.w3.org/1999/xhtml"><head></head><body>' + q + "</body></html>");
    const Wt = tt ? tt.createHTML(q) : q;
    if (Tt === gt)
      try {
        mt = new G().parseFromString(Wt, Rt);
      } catch {
      }
    if (!mt || !mt.documentElement) {
      mt = it.createDocument(Tt, "template", null);
      try {
        mt.documentElement.innerHTML = kt ? dt : Wt;
      } catch {
      }
    }
    const ve = mt.body || mt.documentElement;
    return q && _t && ve.insertBefore(n.createTextNode(_t), ve.childNodes[0] || null), Tt === gt ? Ct.call(mt, W ? "html" : "body")[0] : W ? mt.documentElement : ve;
  }, yr = function (q) {
    return ht.call(
      q.ownerDocument || q,
      q,
      // eslint-disable-next-line no-bitwise
      E.SHOW_ELEMENT | E.SHOW_COMMENT | E.SHOW_TEXT | E.SHOW_PROCESSING_INSTRUCTION | E.SHOW_CDATA_SECTION,
      null
    );
  }, Re = function (q) {
    return q instanceof H && (typeof q.nodeName != "string" || typeof q.textContent != "string" || typeof q.removeChild != "function" || !(q.attributes instanceof j) || typeof q.removeAttribute != "function" || typeof q.setAttribute != "function" || typeof q.namespaceURI != "string" || typeof q.insertBefore != "function" || typeof q.hasChildNodes != "function");
  }, ba = function (q) {
    return typeof _ == "function" && q instanceof _;
  }, $ = function (q, mt, _t) {
    Et[q] && Ca(Et[q], (Wt) => {
      Wt.call(e, mt, _t, ie);
    });
  }, bt = function (q) {
    let mt = null;
    if ($("beforeSanitizeElements", q, null), Re(q))
      return Ve(q), !0;
    const _t = zt(q.nodeName);
    if ($("uponSanitizeElement", q, {
      tagName: _t,
      allowedTags: u
    }), q.hasChildNodes() && !ba(q.firstElementChild) && Ce(/<[/\w]/g, q.innerHTML) && Ce(/<[/\w]/g, q.textContent) || q.nodeType === Sr.progressingInstruction || N && q.nodeType === Sr.comment && Ce(/<[/\w]/g, q.data))
      return Ve(q), !0;
    if (!u[_t] || x[_t]) {
      if (!x[_t] && Pt(_t) && (y.tagNameCheck instanceof RegExp && Ce(y.tagNameCheck, _t) || y.tagNameCheck instanceof Function && y.tagNameCheck(_t)))
        return !1;
      if (Z && !et[_t]) {
        const Wt = A(q) || q.parentNode, ve = L(q) || q.childNodes;
        if (ve && Wt) {
          const de = ve.length;
          for (let Pe = de - 1; Pe >= 0; --Pe) {
            const bn = rt(ve[Pe], !0);
            bn.__removalCount = (q.__removalCount || 0) + 1, Wt.insertBefore(bn, Y(q));
          }
        }
      }
      return Ve(q), !0;
    }
    return q instanceof w && !ns(q) || (_t === "noscript" || _t === "noembed" || _t === "noframes") && Ce(/<\/no(script|embed|frames)/i, q.innerHTML) ? (Ve(q), !0) : (O && q.nodeType === Sr.text && (mt = q.textContent, Ca([Ht, It, l], (Wt) => {
      mt = _r(mt, Wt, " ");
    }), q.textContent !== mt && (xr(e.removed, {
      element: q.cloneNode()
    }), q.textContent = mt)), $("afterSanitizeElements", q, null), !1);
  }, Ot = function (q, mt, _t) {
    if (K && (mt === "id" || mt === "name") && (_t in n || _t in fi))
      return !1;
    if (!(D && !S[mt] && Ce(a, mt))) {
      if (!(k && Ce(c, mt))) {
        if (!p[mt] || S[mt]) {
          if (
            // First condition does a very basic check if a) it's basically a valid custom element tagname AND
            // b) if the tagName passes whatever the user has configured for CUSTOM_ELEMENT_HANDLING.tagNameCheck
            // and c) if the attribute name passes whatever the user has configured for CUSTOM_ELEMENT_HANDLING.attributeNameCheck
            !(Pt(q) && (y.tagNameCheck instanceof RegExp && Ce(y.tagNameCheck, q) || y.tagNameCheck instanceof Function && y.tagNameCheck(q)) && (y.attributeNameCheck instanceof RegExp && Ce(y.attributeNameCheck, mt) || y.attributeNameCheck instanceof Function && y.attributeNameCheck(mt)) || // Alternative, second condition checks if it's an `is`-attribute, AND
              // the value passes whatever the user has configured for CUSTOM_ELEMENT_HANDLING.tagNameCheck
              mt === "is" && y.allowCustomizedBuiltInElements && (y.tagNameCheck instanceof RegExp && Ce(y.tagNameCheck, _t) || y.tagNameCheck instanceof Function && y.tagNameCheck(_t)))
          ) return !1;
        } else if (!ot[mt]) {
          if (!Ce(s, _r(_t, i, ""))) {
            if (!((mt === "src" || mt === "xlink:href" || mt === "href") && q !== "script" && Bg(_t, "data:") === 0 && at[q])) {
              if (!(M && !Ce(f, _r(_t, i, "")))) {
                if (_t)
                  return !1;
              }
            }
          }
        }
      }
    }
    return !0;
  }, Pt = function (q) {
    return q !== "annotation-xml" && bu(q, r);
  }, Mt = function (q) {
    $("beforeSanitizeAttributes", q, null);
    const {
      attributes: mt
    } = q;
    if (!mt)
      return;
    const _t = {
      attrName: "",
      attrValue: "",
      keepAttr: !0,
      allowedAttributes: p
    };
    let Wt = mt.length;
    for (; Wt--;) {
      const ve = mt[Wt], {
        name: de,
        namespaceURI: Pe,
        value: bn
      } = ve, br = zt(de);
      let we = de === "value" ? bn : zg(bn);
      if (_t.attrName = br, _t.attrValue = we, _t.keepAttr = !0, _t.forceKeepAttr = void 0, $("uponSanitizeAttribute", q, _t), we = _t.attrValue, N && Ce(/((--!?|])>)|<\/(style|title)/i, we)) {
        Ge(de, q);
        continue;
      }
      if (_t.forceKeepAttr || (Ge(de, q), !_t.keepAttr))
        continue;
      if (!C && Ce(/\/>/i, we)) {
        Ge(de, q);
        continue;
      }
      O && Ca([Ht, It, l], (Yc) => {
        we = _r(we, Yc, " ");
      });
      const Xc = zt(q.nodeName);
      if (Ot(Xc, br, we)) {
        if (U && (br === "id" || br === "name") && (Ge(de, q), we = B + we), tt && typeof J == "object" && typeof J.getAttributeType == "function" && !Pe)
          switch (J.getAttributeType(Xc, br)) {
            case "TrustedHTML": {
              we = tt.createHTML(we);
              break;
            }
            case "TrustedScriptURL": {
              we = tt.createScriptURL(we);
              break;
            }
          }
        try {
          Pe ? q.setAttributeNS(Pe, de, we) : q.setAttribute(de, we), Re(q) ? Ve(q) : yu(e.removed);
        } catch {
        }
      }
    }
    $("afterSanitizeAttributes", q, null);
  }, Zt = function wt(q) {
    let mt = null;
    const _t = yr(q);
    for ($("beforeSanitizeShadowDOM", q, null); mt = _t.nextNode();)
      $("uponSanitizeShadowNode", mt, null), !bt(mt) && (mt.content instanceof g && wt(mt.content), Mt(mt));
    $("afterSanitizeShadowDOM", q, null);
  };
  return e.sanitize = function (wt) {
    let q = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, mt = null, _t = null, Wt = null, ve = null;
    if (kt = !wt, kt && (wt = "<!-->"), typeof wt != "string" && !ba(wt))
      if (typeof wt.toString == "function") {
        if (wt = wt.toString(), typeof wt != "string")
          throw wr("dirty is not a string, aborting");
      } else
        throw wr("toString is not a function");
    if (!e.isSupported)
      return wt;
    if (X || vr(q), e.removed = [], typeof wt == "string" && (z = !1), z) {
      if (wt.nodeName) {
        const bn = zt(wt.nodeName);
        if (!u[bn] || x[bn])
          throw wr("root node is forbidden and cannot be sanitized in-place");
      }
    } else if (wt instanceof _)
      mt = ya("<!---->"), _t = mt.ownerDocument.importNode(wt, !0), _t.nodeType === Sr.element && _t.nodeName === "BODY" || _t.nodeName === "HTML" ? mt = _t : mt.appendChild(_t);
    else {
      if (!T && !O && !W && // eslint-disable-next-line unicorn/prefer-includes
        wt.indexOf("<") === -1)
        return tt && Q ? tt.createHTML(wt) : wt;
      if (mt = ya(wt), !mt)
        return T ? null : Q ? dt : "";
    }
    mt && I && Ve(mt.firstChild);
    const de = yr(z ? wt : mt);
    for (; Wt = de.nextNode();)
      bt(Wt) || (Wt.content instanceof g && Zt(Wt.content), Mt(Wt));
    if (z)
      return wt;
    if (T) {
      if (R)
        for (ve = v.call(mt.ownerDocument); mt.firstChild;)
          ve.appendChild(mt.firstChild);
      else
        ve = mt;
      return (p.shadowroot || p.shadowrootmode) && (ve = F.call(o, ve, !0)), ve;
    }
    let Pe = W ? mt.outerHTML : mt.innerHTML;
    return W && u["!doctype"] && mt.ownerDocument && mt.ownerDocument.doctype && mt.ownerDocument.doctype.name && Ce(Cf, mt.ownerDocument.doctype.name) && (Pe = "<!DOCTYPE " + mt.ownerDocument.doctype.name + `>
` + Pe), O && Ca([Ht, It, l], (bn) => {
      Pe = _r(Pe, bn, " ");
    }), tt && Q ? tt.createHTML(Pe) : Pe;
  }, e.setConfig = function () {
    let wt = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
    vr(wt), X = !0;
  }, e.clearConfig = function () {
    ie = null, X = !1;
  }, e.isValidAttribute = function (wt, q, mt) {
    ie || vr({});
    const _t = zt(wt), Wt = zt(q);
    return Ot(_t, Wt, mt);
  }, e.addHook = function (wt, q) {
    typeof q == "function" && (Et[wt] = Et[wt] || [], xr(Et[wt], q));
  }, e.removeHook = function (wt) {
    if (Et[wt])
      return yu(Et[wt]);
  }, e.removeHooks = function (wt) {
    Et[wt] && (Et[wt] = []);
  }, e.removeAllHooks = function () {
    Et = {};
  }, e;
}
Sf();
(function () {
  try {
    if (typeof document < "u") {
      var t = document.createElement("style");
      t.appendChild(document.createTextNode(":root{--svg-filter: #313131;--svg-filter-active: #0095ff;--font-color: #000;--background-color: #fff;--background-color-active: #f0f0f0}[data-theme=dark]{--svg-filter: white;--svg-filter-active: #a5ffff;--font-color: #fff;--background-color: #333;--background-color-active: #7a7a7a}.panel-paint-box{width:100%;height:100%}#prompt-input-box{display:flex;justify-content:center;width:80%;margin-inline:auto}#prompt-input-box #prompt-input{border-radius:1rem;border-style:solid;border-width:1px;width:80%;padding:.2rem 1rem;margin-inline:.3rem;color:var(--font-color);background-color:var(--background-color)}#prompt-input-box #wand-icon{transition:color 1s ease}.blinking{animation:blink .5s infinite alternate}@keyframes blink{0%{opacity:1}to{opacity:.1}}.shining{fill:#ffcd28!important;animation:shine .5s forwards}@keyframes shine{0%{transform:scale(1)}50%{transform:scale(1.5)}to{transform:scale(1)}}#tool-bar{display:flex;width:100%;overflow-x:scroll;scrollbar-width:none;-ms-overflow-style:none}#tool-bar::-webkit-scrollbar{display:none}#essential-tool-box{display:flex}.painter-tool-icon{width:1.5rem;height:1.5rem;margin-inline:.2rem;cursor:pointer;display:flex;align-self:center}.painter-tool-icon svg{width:100%;height:100%;align-self:center;-webkit-user-drag:none;-khtml-user-drag:none;-moz-user-drag:none;-o-user-drag:none}.painter-tool-icon svg:hover{opacity:.8}.painter-tool-icon svg{fill:var(--svg-filter)}.painter-tool-icon svg.active{fill:var(--svg-filter-active)}.painter-tool-icon.disabled{opacity:.3;cursor:not-allowed}#stroke-size-box{display:flex}#stroke-size-box #stroke-width-slider{width:6rem;align-self:center}#stroke-size-box #stroke-width-value{font-size:.8rem;margin-left:.2rem;align-self:center;color:var(--font-color)}#stroke-color{align-self:center;width:3rem;height:1.5rem;padding:0}#stroke-color-transparent{align-self:center;width:3.5rem;height:1.5rem;padding:.2rem;margin-left:.2rem;font-size:.8rem;color:var(--font-color);background-color:var(--background-color);border-radius:1rem;border-style:solid;border-width:1px;border-radius:.2rem;box-sizing:border-box}#painter-history-panel{display:flex}#download-box{margin-left:auto}.side-bar{width:7rem;display:flex;flex-direction:column;justify-content:space-between}.side-bar button{width:24px}.top-bar{width:100%;min-height:5rem;display:flex;flex-direction:column;justify-content:space-evenly}.lower-area{width:100%;height:calc(100% - 7rem);display:flex}.separator{align-self:center;width:1px;height:25px;background-color:var(--border-color);display:inline-block;vertical-align:middle;margin:0 4px}#color-and-alpha-box{display:flex}#output-img-box{position:relative}#output-operation-box{position:absolute;bottom:.5rem;width:80%;left:10%;background:#e3e3e333;padding:.5rem 1rem;box-sizing:border-box;border-radius:.5rem;display:flex;justify-content:space-around}.side-bar .layer-box{display:flex;flex-direction:column;height:calc(100% - 1.5rem)}.layer-item-list{display:flex;flex-direction:column;gap:2px;align-items:center;overflow-y:auto;overflow-x:hidden;padding-block:.5rem;max-height:350px}.layer-operation-box{display:flex;justify-content:space-between;padding-inline:.3rem}.layer-item-container{flex-direction:row;display:flex;justify-content:space-between;align-items:center;width:85%;padding:.05rem .4rem;margin-block:.1rem;box-sizing:border-box;border-radius:.4rem;border:1px solid #737373}.layer-item-container .layer-item{width:80%;font-size:.7rem;color:var(--font-color);cursor:pointer}.layer-item-container:hover{opacity:.8}.layer-item-container.active{background:var(--background-color-active)}.layer-item-container .layer-item-remove{width:20%;border-radius:50%;box-sizing:border-box;padding:2%;cursor:pointer;fill:var(--svg-filter);-webkit-user-drag:none;-khtml-user-drag:none;-moz-user-drag:none;-o-user-drag:none}.layer-item-container .layer-item-remove:hover{opacity:.6}#output-area{width:calc(100% - 7rem);display:flex}#canvas-box{width:50%}#left-box{position:relative}#bg-reminder{position:absolute;width:100%;height:100%;z-index:10;display:flex;cursor:pointer}#bg-reminder-text{color:#fff;text-align:center;display:block;margin:auto;padding:.3rem 1rem;border-radius:1rem}#output-img{background-color:gray;width:100%;height:100%}#loading-status{text-align:center;margin-top:1rem;height:1.5rem}#loading-status svg{width:1.2rem;height:1.2rem;margin:auto;fill:var(--svg-filter)}@-webkit-keyframes rotating{0%{-webkit-transform:rotate(0deg);-o-transform:rotate(0deg);transform:rotate(0)}to{-webkit-transform:rotate(360deg);-o-transform:rotate(360deg);transform:rotate(360deg)}}@keyframes rotating{0%{-ms-transform:rotate(0deg);-moz-transform:rotate(0deg);-webkit-transform:rotate(0deg);-o-transform:rotate(0deg);transform:rotate(0)}to{-ms-transform:rotate(360deg);-moz-transform:rotate(360deg);-webkit-transform:rotate(360deg);-o-transform:rotate(360deg);transform:rotate(360deg)}}.rotating{-webkit-animation:rotating 2s linear infinite;-moz-animation:rotating 2s linear infinite;-ms-animation:rotating 2s linear infinite;-o-animation:rotating 2s linear infinite;animation:rotating 2s linear infinite}")), document.head.appendChild(t);
    }
  } catch (e) {
    console.error("vite-plugin-css-injected-by-js", e);
  }
})();
function Vl(t) {
  return t && t.__esModule && Object.prototype.hasOwnProperty.call(t, "default") ? t.default : t;
}
var Ef = { exports: {} }, zo = {}, kf = { exports: {} }, Ft = {};
/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var ha = Symbol.for("react.element"), tm = Symbol.for("react.portal"), em = Symbol.for("react.fragment"), nm = Symbol.for("react.strict_mode"), im = Symbol.for("react.profiler"), rm = Symbol.for("react.provider"), am = Symbol.for("react.context"), om = Symbol.for("react.forward_ref"), sm = Symbol.for("react.suspense"), lm = Symbol.for("react.memo"), cm = Symbol.for("react.lazy"), Eu = Symbol.iterator;
function um(t) {
  return t === null || typeof t != "object" ? null : (t = Eu && t[Eu] || t["@@iterator"], typeof t == "function" ? t : null);
}
var Tf = {
  isMounted: function () {
    return !1;
  }, enqueueForceUpdate: function () {
  }, enqueueReplaceState: function () {
  }, enqueueSetState: function () {
  }
}, Of = Object.assign, Pf = {};
function dr(t, e, n) {
  this.props = t, this.context = e, this.refs = Pf, this.updater = n || Tf;
}
dr.prototype.isReactComponent = {};
dr.prototype.setState = function (t, e) {
  if (typeof t != "object" && typeof t != "function" && t != null) throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
  this.updater.enqueueSetState(this, t, e, "setState");
};
dr.prototype.forceUpdate = function (t) {
  this.updater.enqueueForceUpdate(this, t, "forceUpdate");
};
function Mf() {
}
Mf.prototype = dr.prototype;
function Gl(t, e, n) {
  this.props = t, this.context = e, this.refs = Pf, this.updater = n || Tf;
}
var ql = Gl.prototype = new Mf();
ql.constructor = Gl;
Of(ql, dr.prototype);
ql.isPureReactComponent = !0;
var ku = Array.isArray, Af = Object.prototype.hasOwnProperty, Zl = { current: null }, Df = { key: !0, ref: !0, __self: !0, __source: !0 };
function jf(t, e, n) {
  var o, d = {}, g = null, b = null;
  if (e != null) for (o in e.ref !== void 0 && (b = e.ref), e.key !== void 0 && (g = "" + e.key), e) Af.call(e, o) && !Df.hasOwnProperty(o) && (d[o] = e[o]);
  var _ = arguments.length - 2;
  if (_ === 1) d.children = n;
  else if (1 < _) {
    for (var w = Array(_), E = 0; E < _; E++) w[E] = arguments[E + 2];
    d.children = w;
  }
  if (t && t.defaultProps) for (o in _ = t.defaultProps, _) d[o] === void 0 && (d[o] = _[o]);
  return { $$typeof: ha, type: t, key: g, ref: b, props: d, _owner: Zl.current };
}
function hm(t, e) {
  return { $$typeof: ha, type: t.type, key: e, ref: t.ref, props: t.props, _owner: t._owner };
}
function Ql(t) {
  return typeof t == "object" && t !== null && t.$$typeof === ha;
}
function fm(t) {
  var e = { "=": "=0", ":": "=2" };
  return "$" + t.replace(/[=:]/g, function (n) {
    return e[n];
  });
}
var Tu = /\/+/g;
function hs(t, e) {
  return typeof t == "object" && t !== null && t.key != null ? fm("" + t.key) : e.toString(36);
}
function Za(t, e, n, o, d) {
  var g = typeof t;
  (g === "undefined" || g === "boolean") && (t = null);
  var b = !1;
  if (t === null) b = !0;
  else switch (g) {
    case "string":
    case "number":
      b = !0;
      break;
    case "object":
      switch (t.$$typeof) {
        case ha:
        case tm:
          b = !0;
      }
  }
  if (b) return b = t, d = d(b), t = o === "" ? "." + hs(b, 0) : o, ku(d) ? (n = "", t != null && (n = t.replace(Tu, "$&/") + "/"), Za(d, e, n, "", function (E) {
    return E;
  })) : d != null && (Ql(d) && (d = hm(d, n + (!d.key || b && b.key === d.key ? "" : ("" + d.key).replace(Tu, "$&/") + "/") + t)), e.push(d)), 1;
  if (b = 0, o = o === "" ? "." : o + ":", ku(t)) for (var _ = 0; _ < t.length; _++) {
    g = t[_];
    var w = o + hs(g, _);
    b += Za(g, e, n, w, d);
  }
  else if (w = um(t), typeof w == "function") for (t = w.call(t), _ = 0; !(g = t.next()).done;) g = g.value, w = o + hs(g, _++), b += Za(g, e, n, w, d);
  else if (g === "object") throw e = String(t), Error("Objects are not valid as a React child (found: " + (e === "[object Object]" ? "object with keys {" + Object.keys(t).join(", ") + "}" : e) + "). If you meant to render a collection of children, use an array instead.");
  return b;
}
function Ea(t, e, n) {
  if (t == null) return t;
  var o = [], d = 0;
  return Za(t, o, "", "", function (g) {
    return e.call(n, g, d++);
  }), o;
}
function dm(t) {
  if (t._status === -1) {
    var e = t._result;
    e = e(), e.then(function (n) {
      (t._status === 0 || t._status === -1) && (t._status = 1, t._result = n);
    }, function (n) {
      (t._status === 0 || t._status === -1) && (t._status = 2, t._result = n);
    }), t._status === -1 && (t._status = 0, t._result = e);
  }
  if (t._status === 1) return t._result.default;
  throw t._result;
}
var Te = { current: null }, Qa = { transition: null }, pm = { ReactCurrentDispatcher: Te, ReactCurrentBatchConfig: Qa, ReactCurrentOwner: Zl };
function Lf() {
  throw Error("act(...) is not supported in production builds of React.");
}
Ft.Children = {
  map: Ea, forEach: function (t, e, n) {
    Ea(t, function () {
      e.apply(this, arguments);
    }, n);
  }, count: function (t) {
    var e = 0;
    return Ea(t, function () {
      e++;
    }), e;
  }, toArray: function (t) {
    return Ea(t, function (e) {
      return e;
    }) || [];
  }, only: function (t) {
    if (!Ql(t)) throw Error("React.Children.only expected to receive a single React element child.");
    return t;
  }
};
Ft.Component = dr;
Ft.Fragment = em;
Ft.Profiler = im;
Ft.PureComponent = Gl;
Ft.StrictMode = nm;
Ft.Suspense = sm;
Ft.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = pm;
Ft.act = Lf;
Ft.cloneElement = function (t, e, n) {
  if (t == null) throw Error("React.cloneElement(...): The argument must be a React element, but you passed " + t + ".");
  var o = Of({}, t.props), d = t.key, g = t.ref, b = t._owner;
  if (e != null) {
    if (e.ref !== void 0 && (g = e.ref, b = Zl.current), e.key !== void 0 && (d = "" + e.key), t.type && t.type.defaultProps) var _ = t.type.defaultProps;
    for (w in e) Af.call(e, w) && !Df.hasOwnProperty(w) && (o[w] = e[w] === void 0 && _ !== void 0 ? _[w] : e[w]);
  }
  var w = arguments.length - 2;
  if (w === 1) o.children = n;
  else if (1 < w) {
    _ = Array(w);
    for (var E = 0; E < w; E++) _[E] = arguments[E + 2];
    o.children = _;
  }
  return { $$typeof: ha, type: t.type, key: d, ref: g, props: o, _owner: b };
};
Ft.createContext = function (t) {
  return t = { $$typeof: am, _currentValue: t, _currentValue2: t, _threadCount: 0, Provider: null, Consumer: null, _defaultValue: null, _globalName: null }, t.Provider = { $$typeof: rm, _context: t }, t.Consumer = t;
};
Ft.createElement = jf;
Ft.createFactory = function (t) {
  var e = jf.bind(null, t);
  return e.type = t, e;
};
Ft.createRef = function () {
  return { current: null };
};
Ft.forwardRef = function (t) {
  return { $$typeof: om, render: t };
};
Ft.isValidElement = Ql;
Ft.lazy = function (t) {
  return { $$typeof: cm, _payload: { _status: -1, _result: t }, _init: dm };
};
Ft.memo = function (t, e) {
  return { $$typeof: lm, type: t, compare: e === void 0 ? null : e };
};
Ft.startTransition = function (t) {
  var e = Qa.transition;
  Qa.transition = {};
  try {
    t();
  } finally {
    Qa.transition = e;
  }
};
Ft.unstable_act = Lf;
Ft.useCallback = function (t, e) {
  return Te.current.useCallback(t, e);
};
Ft.useContext = function (t) {
  return Te.current.useContext(t);
};
Ft.useDebugValue = function () {
};
Ft.useDeferredValue = function (t) {
  return Te.current.useDeferredValue(t);
};
Ft.useEffect = function (t, e) {
  return Te.current.useEffect(t, e);
};
Ft.useId = function () {
  return Te.current.useId();
};
Ft.useImperativeHandle = function (t, e, n) {
  return Te.current.useImperativeHandle(t, e, n);
};
Ft.useInsertionEffect = function (t, e) {
  return Te.current.useInsertionEffect(t, e);
};
Ft.useLayoutEffect = function (t, e) {
  return Te.current.useLayoutEffect(t, e);
};
Ft.useMemo = function (t, e) {
  return Te.current.useMemo(t, e);
};
Ft.useReducer = function (t, e, n) {
  return Te.current.useReducer(t, e, n);
};
Ft.useRef = function (t) {
  return Te.current.useRef(t);
};
Ft.useState = function (t) {
  return Te.current.useState(t);
};
Ft.useSyncExternalStore = function (t, e, n) {
  return Te.current.useSyncExternalStore(t, e, n);
};
Ft.useTransition = function () {
  return Te.current.useTransition();
};
Ft.version = "18.3.1";
kf.exports = Ft;
var P = kf.exports;
const Ff = /* @__PURE__ */ Vl(P);
/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var gm = P, mm = Symbol.for("react.element"), vm = Symbol.for("react.fragment"), ym = Object.prototype.hasOwnProperty, bm = gm.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, xm = { key: !0, ref: !0, __self: !0, __source: !0 };
function If(t, e, n) {
  var o, d = {}, g = null, b = null;
  n !== void 0 && (g = "" + n), e.key !== void 0 && (g = "" + e.key), e.ref !== void 0 && (b = e.ref);
  for (o in e) ym.call(e, o) && !xm.hasOwnProperty(o) && (d[o] = e[o]);
  if (t && t.defaultProps) for (o in e = t.defaultProps, e) d[o] === void 0 && (d[o] = e[o]);
  return { $$typeof: mm, type: t, key: g, ref: b, props: d, _owner: bm.current };
}
zo.Fragment = vm;
zo.jsx = If;
zo.jsxs = If;
Ef.exports = zo;
var St = Ef.exports, Rf = { exports: {} }, Xe = {}, Bf = { exports: {} }, zf = {};
/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
(function (t) {
  function e(i, r) {
    var s = i.length;
    i.push(r);
    t: for (; 0 < s;) {
      var u = s - 1 >>> 1, h = i[u];
      if (0 < d(h, r)) i[u] = r, i[s] = h, s = u;
      else break t;
    }
  }
  function n(i) {
    return i.length === 0 ? null : i[0];
  }
  function o(i) {
    if (i.length === 0) return null;
    var r = i[0], s = i.pop();
    if (s !== r) {
      i[0] = s;
      t: for (var u = 0, h = i.length, p = h >>> 1; u < p;) {
        var m = 2 * (u + 1) - 1, y = i[m], x = m + 1, S = i[x];
        if (0 > d(y, s)) x < h && 0 > d(S, y) ? (i[u] = S, i[x] = s, u = x) : (i[u] = y, i[m] = s, u = m);
        else if (x < h && 0 > d(S, s)) i[u] = S, i[x] = s, u = x;
        else break t;
      }
    }
    return r;
  }
  function d(i, r) {
    var s = i.sortIndex - r.sortIndex;
    return s !== 0 ? s : i.id - r.id;
  }
  if (typeof performance == "object" && typeof performance.now == "function") {
    var g = performance;
    t.unstable_now = function () {
      return g.now();
    };
  } else {
    var b = Date, _ = b.now();
    t.unstable_now = function () {
      return b.now() - _;
    };
  }
  var w = [], E = [], j = 1, H = null, G = 3, J = !1, nt = !1, rt = !1, vt = typeof setTimeout == "function" ? setTimeout : null, Y = typeof clearTimeout == "function" ? clearTimeout : null, L = typeof setImmediate < "u" ? setImmediate : null;
  typeof navigator < "u" && navigator.scheduling !== void 0 && navigator.scheduling.isInputPending !== void 0 && navigator.scheduling.isInputPending.bind(navigator.scheduling);
  function A(i) {
    for (var r = n(E); r !== null;) {
      if (r.callback === null) o(E);
      else if (r.startTime <= i) o(E), r.sortIndex = r.expirationTime, e(w, r);
      else break;
      r = n(E);
    }
  }
  function tt(i) {
    if (rt = !1, A(i), !nt) if (n(w) !== null) nt = !0, c(dt);
    else {
      var r = n(E);
      r !== null && f(tt, r.startTime - i);
    }
  }
  function dt(i, r) {
    nt = !1, rt && (rt = !1, Y(v), v = -1), J = !0;
    var s = G;
    try {
      for (A(r), H = n(w); H !== null && (!(H.expirationTime > r) || i && !Et());) {
        var u = H.callback;
        if (typeof u == "function") {
          H.callback = null, G = H.priorityLevel;
          var h = u(H.expirationTime <= r);
          r = t.unstable_now(), typeof h == "function" ? H.callback = h : H === n(w) && o(w), A(r);
        } else o(w);
        H = n(w);
      }
      if (H !== null) var p = !0;
      else {
        var m = n(E);
        m !== null && f(tt, m.startTime - r), p = !1;
      }
      return p;
    } finally {
      H = null, G = s, J = !1;
    }
  }
  var it = !1, ht = null, v = -1, Ct = 5, F = -1;
  function Et() {
    return !(t.unstable_now() - F < Ct);
  }
  function Ht() {
    if (ht !== null) {
      var i = t.unstable_now();
      F = i;
      var r = !0;
      try {
        r = ht(!0, i);
      } finally {
        r ? It() : (it = !1, ht = null);
      }
    } else it = !1;
  }
  var It;
  if (typeof L == "function") It = function () {
    L(Ht);
  };
  else if (typeof MessageChannel < "u") {
    var l = new MessageChannel(), a = l.port2;
    l.port1.onmessage = Ht, It = function () {
      a.postMessage(null);
    };
  } else It = function () {
    vt(Ht, 0);
  };
  function c(i) {
    ht = i, it || (it = !0, It());
  }
  function f(i, r) {
    v = vt(function () {
      i(t.unstable_now());
    }, r);
  }
  t.unstable_IdlePriority = 5, t.unstable_ImmediatePriority = 1, t.unstable_LowPriority = 4, t.unstable_NormalPriority = 3, t.unstable_Profiling = null, t.unstable_UserBlockingPriority = 2, t.unstable_cancelCallback = function (i) {
    i.callback = null;
  }, t.unstable_continueExecution = function () {
    nt || J || (nt = !0, c(dt));
  }, t.unstable_forceFrameRate = function (i) {
    0 > i || 125 < i ? console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported") : Ct = 0 < i ? Math.floor(1e3 / i) : 5;
  }, t.unstable_getCurrentPriorityLevel = function () {
    return G;
  }, t.unstable_getFirstCallbackNode = function () {
    return n(w);
  }, t.unstable_next = function (i) {
    switch (G) {
      case 1:
      case 2:
      case 3:
        var r = 3;
        break;
      default:
        r = G;
    }
    var s = G;
    G = r;
    try {
      return i();
    } finally {
      G = s;
    }
  }, t.unstable_pauseExecution = function () {
  }, t.unstable_requestPaint = function () {
  }, t.unstable_runWithPriority = function (i, r) {
    switch (i) {
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
        break;
      default:
        i = 3;
    }
    var s = G;
    G = i;
    try {
      return r();
    } finally {
      G = s;
    }
  }, t.unstable_scheduleCallback = function (i, r, s) {
    var u = t.unstable_now();
    switch (typeof s == "object" && s !== null ? (s = s.delay, s = typeof s == "number" && 0 < s ? u + s : u) : s = u, i) {
      case 1:
        var h = -1;
        break;
      case 2:
        h = 250;
        break;
      case 5:
        h = 1073741823;
        break;
      case 4:
        h = 1e4;
        break;
      default:
        h = 5e3;
    }
    return h = s + h, i = { id: j++, callback: r, priorityLevel: i, startTime: s, expirationTime: h, sortIndex: -1 }, s > u ? (i.sortIndex = s, e(E, i), n(w) === null && i === n(E) && (rt ? (Y(v), v = -1) : rt = !0, f(tt, s - u))) : (i.sortIndex = h, e(w, i), nt || J || (nt = !0, c(dt))), i;
  }, t.unstable_shouldYield = Et, t.unstable_wrapCallback = function (i) {
    var r = G;
    return function () {
      var s = G;
      G = r;
      try {
        return i.apply(this, arguments);
      } finally {
        G = s;
      }
    };
  };
})(zf);
Bf.exports = zf;
var _m = Bf.exports;
/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var wm = P, We = _m;
function pt(t) {
  for (var e = "https://reactjs.org/docs/error-decoder.html?invariant=" + t, n = 1; n < arguments.length; n++) e += "&args[]=" + encodeURIComponent(arguments[n]);
  return "Minified React error #" + t + "; visit " + e + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
}
var Nf = /* @__PURE__ */ new Set(), Gr = {};
function Ai(t, e) {
  rr(t, e), rr(t + "Capture", e);
}
function rr(t, e) {
  for (Gr[t] = e, t = 0; t < e.length; t++) Nf.add(e[t]);
}
var Nn = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), Zs = Object.prototype.hasOwnProperty, Cm = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/, Ou = {}, Pu = {};
function Sm(t) {
  return Zs.call(Pu, t) ? !0 : Zs.call(Ou, t) ? !1 : Cm.test(t) ? Pu[t] = !0 : (Ou[t] = !0, !1);
}
function Em(t, e, n, o) {
  if (n !== null && n.type === 0) return !1;
  switch (typeof e) {
    case "function":
    case "symbol":
      return !0;
    case "boolean":
      return o ? !1 : n !== null ? !n.acceptsBooleans : (t = t.toLowerCase().slice(0, 5), t !== "data-" && t !== "aria-");
    default:
      return !1;
  }
}
function km(t, e, n, o) {
  if (e === null || typeof e > "u" || Em(t, e, n, o)) return !0;
  if (o) return !1;
  if (n !== null) switch (n.type) {
    case 3:
      return !e;
    case 4:
      return e === !1;
    case 5:
      return isNaN(e);
    case 6:
      return isNaN(e) || 1 > e;
  }
  return !1;
}
function Oe(t, e, n, o, d, g, b) {
  this.acceptsBooleans = e === 2 || e === 3 || e === 4, this.attributeName = o, this.attributeNamespace = d, this.mustUseProperty = n, this.propertyName = t, this.type = e, this.sanitizeURL = g, this.removeEmptyString = b;
}
var me = {};
"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function (t) {
  me[t] = new Oe(t, 0, !1, t, null, !1, !1);
});
[["acceptCharset", "accept-charset"], ["className", "class"], ["htmlFor", "for"], ["httpEquiv", "http-equiv"]].forEach(function (t) {
  var e = t[0];
  me[e] = new Oe(e, 1, !1, t[1], null, !1, !1);
});
["contentEditable", "draggable", "spellCheck", "value"].forEach(function (t) {
  me[t] = new Oe(t, 2, !1, t.toLowerCase(), null, !1, !1);
});
["autoReverse", "externalResourcesRequired", "focusable", "preserveAlpha"].forEach(function (t) {
  me[t] = new Oe(t, 2, !1, t, null, !1, !1);
});
"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function (t) {
  me[t] = new Oe(t, 3, !1, t.toLowerCase(), null, !1, !1);
});
["checked", "multiple", "muted", "selected"].forEach(function (t) {
  me[t] = new Oe(t, 3, !0, t, null, !1, !1);
});
["capture", "download"].forEach(function (t) {
  me[t] = new Oe(t, 4, !1, t, null, !1, !1);
});
["cols", "rows", "size", "span"].forEach(function (t) {
  me[t] = new Oe(t, 6, !1, t, null, !1, !1);
});
["rowSpan", "start"].forEach(function (t) {
  me[t] = new Oe(t, 5, !1, t.toLowerCase(), null, !1, !1);
});
var Kl = /[\-:]([a-z])/g;
function Jl(t) {
  return t[1].toUpperCase();
}
"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function (t) {
  var e = t.replace(
    Kl,
    Jl
  );
  me[e] = new Oe(e, 1, !1, t, null, !1, !1);
});
"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function (t) {
  var e = t.replace(Kl, Jl);
  me[e] = new Oe(e, 1, !1, t, "http://www.w3.org/1999/xlink", !1, !1);
});
["xml:base", "xml:lang", "xml:space"].forEach(function (t) {
  var e = t.replace(Kl, Jl);
  me[e] = new Oe(e, 1, !1, t, "http://www.w3.org/XML/1998/namespace", !1, !1);
});
["tabIndex", "crossOrigin"].forEach(function (t) {
  me[t] = new Oe(t, 1, !1, t.toLowerCase(), null, !1, !1);
});
me.xlinkHref = new Oe("xlinkHref", 1, !1, "xlink:href", "http://www.w3.org/1999/xlink", !0, !1);
["src", "href", "action", "formAction"].forEach(function (t) {
  me[t] = new Oe(t, 1, !1, t.toLowerCase(), null, !0, !0);
});
function $l(t, e, n, o) {
  var d = me.hasOwnProperty(e) ? me[e] : null;
  (d !== null ? d.type !== 0 : o || !(2 < e.length) || e[0] !== "o" && e[0] !== "O" || e[1] !== "n" && e[1] !== "N") && (km(e, n, d, o) && (n = null), o || d === null ? Sm(e) && (n === null ? t.removeAttribute(e) : t.setAttribute(e, "" + n)) : d.mustUseProperty ? t[d.propertyName] = n === null ? d.type === 3 ? !1 : "" : n : (e = d.attributeName, o = d.attributeNamespace, n === null ? t.removeAttribute(e) : (d = d.type, n = d === 3 || d === 4 && n === !0 ? "" : "" + n, o ? t.setAttributeNS(o, e, n) : t.setAttribute(e, n))));
}
var Xn = wm.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED, ka = Symbol.for("react.element"), zi = Symbol.for("react.portal"), Ni = Symbol.for("react.fragment"), tc = Symbol.for("react.strict_mode"), Qs = Symbol.for("react.profiler"), Uf = Symbol.for("react.provider"), Hf = Symbol.for("react.context"), ec = Symbol.for("react.forward_ref"), Ks = Symbol.for("react.suspense"), Js = Symbol.for("react.suspense_list"), nc = Symbol.for("react.memo"), Vn = Symbol.for("react.lazy"), Wf = Symbol.for("react.offscreen"), Mu = Symbol.iterator;
function Er(t) {
  return t === null || typeof t != "object" ? null : (t = Mu && t[Mu] || t["@@iterator"], typeof t == "function" ? t : null);
}
var ne = Object.assign, fs;
function jr(t) {
  if (fs === void 0) try {
    throw Error();
  } catch (n) {
    var e = n.stack.trim().match(/\n( *(at )?)/);
    fs = e && e[1] || "";
  }
  return `
` + fs + t;
}
var ds = !1;
function ps(t, e) {
  if (!t || ds) return "";
  ds = !0;
  var n = Error.prepareStackTrace;
  Error.prepareStackTrace = void 0;
  try {
    if (e) if (e = function () {
      throw Error();
    }, Object.defineProperty(e.prototype, "props", {
      set: function () {
        throw Error();
      }
    }), typeof Reflect == "object" && Reflect.construct) {
      try {
        Reflect.construct(e, []);
      } catch (E) {
        var o = E;
      }
      Reflect.construct(t, [], e);
    } else {
      try {
        e.call();
      } catch (E) {
        o = E;
      }
      t.call(e.prototype);
    }
    else {
      try {
        throw Error();
      } catch (E) {
        o = E;
      }
      t();
    }
  } catch (E) {
    if (E && o && typeof E.stack == "string") {
      for (var d = E.stack.split(`
`), g = o.stack.split(`
`), b = d.length - 1, _ = g.length - 1; 1 <= b && 0 <= _ && d[b] !== g[_];) _--;
      for (; 1 <= b && 0 <= _; b--, _--) if (d[b] !== g[_]) {
        if (b !== 1 || _ !== 1)
          do
            if (b--, _--, 0 > _ || d[b] !== g[_]) {
              var w = `
` + d[b].replace(" at new ", " at ");
              return t.displayName && w.includes("<anonymous>") && (w = w.replace("<anonymous>", t.displayName)), w;
            }
          while (1 <= b && 0 <= _);
        break;
      }
    }
  } finally {
    ds = !1, Error.prepareStackTrace = n;
  }
  return (t = t ? t.displayName || t.name : "") ? jr(t) : "";
}
function Tm(t) {
  switch (t.tag) {
    case 5:
      return jr(t.type);
    case 16:
      return jr("Lazy");
    case 13:
      return jr("Suspense");
    case 19:
      return jr("SuspenseList");
    case 0:
    case 2:
    case 15:
      return t = ps(t.type, !1), t;
    case 11:
      return t = ps(t.type.render, !1), t;
    case 1:
      return t = ps(t.type, !0), t;
    default:
      return "";
  }
}
function $s(t) {
  if (t == null) return null;
  if (typeof t == "function") return t.displayName || t.name || null;
  if (typeof t == "string") return t;
  switch (t) {
    case Ni:
      return "Fragment";
    case zi:
      return "Portal";
    case Qs:
      return "Profiler";
    case tc:
      return "StrictMode";
    case Ks:
      return "Suspense";
    case Js:
      return "SuspenseList";
  }
  if (typeof t == "object") switch (t.$$typeof) {
    case Hf:
      return (t.displayName || "Context") + ".Consumer";
    case Uf:
      return (t._context.displayName || "Context") + ".Provider";
    case ec:
      var e = t.render;
      return t = t.displayName, t || (t = e.displayName || e.name || "", t = t !== "" ? "ForwardRef(" + t + ")" : "ForwardRef"), t;
    case nc:
      return e = t.displayName || null, e !== null ? e : $s(t.type) || "Memo";
    case Vn:
      e = t._payload, t = t._init;
      try {
        return $s(t(e));
      } catch {
      }
  }
  return null;
}
function Om(t) {
  var e = t.type;
  switch (t.tag) {
    case 24:
      return "Cache";
    case 9:
      return (e.displayName || "Context") + ".Consumer";
    case 10:
      return (e._context.displayName || "Context") + ".Provider";
    case 18:
      return "DehydratedFragment";
    case 11:
      return t = e.render, t = t.displayName || t.name || "", e.displayName || (t !== "" ? "ForwardRef(" + t + ")" : "ForwardRef");
    case 7:
      return "Fragment";
    case 5:
      return e;
    case 4:
      return "Portal";
    case 3:
      return "Root";
    case 6:
      return "Text";
    case 16:
      return $s(e);
    case 8:
      return e === tc ? "StrictMode" : "Mode";
    case 22:
      return "Offscreen";
    case 12:
      return "Profiler";
    case 21:
      return "Scope";
    case 13:
      return "Suspense";
    case 19:
      return "SuspenseList";
    case 25:
      return "TracingMarker";
    case 1:
    case 0:
    case 17:
    case 2:
    case 14:
    case 15:
      if (typeof e == "function") return e.displayName || e.name || null;
      if (typeof e == "string") return e;
  }
  return null;
}
function si(t) {
  switch (typeof t) {
    case "boolean":
    case "number":
    case "string":
    case "undefined":
      return t;
    case "object":
      return t;
    default:
      return "";
  }
}
function Xf(t) {
  var e = t.type;
  return (t = t.nodeName) && t.toLowerCase() === "input" && (e === "checkbox" || e === "radio");
}
function Pm(t) {
  var e = Xf(t) ? "checked" : "value", n = Object.getOwnPropertyDescriptor(t.constructor.prototype, e), o = "" + t[e];
  if (!t.hasOwnProperty(e) && typeof n < "u" && typeof n.get == "function" && typeof n.set == "function") {
    var d = n.get, g = n.set;
    return Object.defineProperty(t, e, {
      configurable: !0, get: function () {
        return d.call(this);
      }, set: function (b) {
        o = "" + b, g.call(this, b);
      }
    }), Object.defineProperty(t, e, { enumerable: n.enumerable }), {
      getValue: function () {
        return o;
      }, setValue: function (b) {
        o = "" + b;
      }, stopTracking: function () {
        t._valueTracker = null, delete t[e];
      }
    };
  }
}
function Ta(t) {
  t._valueTracker || (t._valueTracker = Pm(t));
}
function Yf(t) {
  if (!t) return !1;
  var e = t._valueTracker;
  if (!e) return !0;
  var n = e.getValue(), o = "";
  return t && (o = Xf(t) ? t.checked ? "true" : "false" : t.value), t = o, t !== n ? (e.setValue(t), !0) : !1;
}
function lo(t) {
  if (t = t || (typeof document < "u" ? document : void 0), typeof t > "u") return null;
  try {
    return t.activeElement || t.body;
  } catch {
    return t.body;
  }
}
function tl(t, e) {
  var n = e.checked;
  return ne({}, e, { defaultChecked: void 0, defaultValue: void 0, value: void 0, checked: n ?? t._wrapperState.initialChecked });
}
function Au(t, e) {
  var n = e.defaultValue == null ? "" : e.defaultValue, o = e.checked != null ? e.checked : e.defaultChecked;
  n = si(e.value != null ? e.value : n), t._wrapperState = { initialChecked: o, initialValue: n, controlled: e.type === "checkbox" || e.type === "radio" ? e.checked != null : e.value != null };
}
function Vf(t, e) {
  e = e.checked, e != null && $l(t, "checked", e, !1);
}
function el(t, e) {
  Vf(t, e);
  var n = si(e.value), o = e.type;
  if (n != null) o === "number" ? (n === 0 && t.value === "" || t.value != n) && (t.value = "" + n) : t.value !== "" + n && (t.value = "" + n);
  else if (o === "submit" || o === "reset") {
    t.removeAttribute("value");
    return;
  }
  e.hasOwnProperty("value") ? nl(t, e.type, n) : e.hasOwnProperty("defaultValue") && nl(t, e.type, si(e.defaultValue)), e.checked == null && e.defaultChecked != null && (t.defaultChecked = !!e.defaultChecked);
}
function Du(t, e, n) {
  if (e.hasOwnProperty("value") || e.hasOwnProperty("defaultValue")) {
    var o = e.type;
    if (!(o !== "submit" && o !== "reset" || e.value !== void 0 && e.value !== null)) return;
    e = "" + t._wrapperState.initialValue, n || e === t.value || (t.value = e), t.defaultValue = e;
  }
  n = t.name, n !== "" && (t.name = ""), t.defaultChecked = !!t._wrapperState.initialChecked, n !== "" && (t.name = n);
}
function nl(t, e, n) {
  (e !== "number" || lo(t.ownerDocument) !== t) && (n == null ? t.defaultValue = "" + t._wrapperState.initialValue : t.defaultValue !== "" + n && (t.defaultValue = "" + n));
}
var Lr = Array.isArray;
function Ji(t, e, n, o) {
  if (t = t.options, e) {
    e = {};
    for (var d = 0; d < n.length; d++) e["$" + n[d]] = !0;
    for (n = 0; n < t.length; n++) d = e.hasOwnProperty("$" + t[n].value), t[n].selected !== d && (t[n].selected = d), d && o && (t[n].defaultSelected = !0);
  } else {
    for (n = "" + si(n), e = null, d = 0; d < t.length; d++) {
      if (t[d].value === n) {
        t[d].selected = !0, o && (t[d].defaultSelected = !0);
        return;
      }
      e !== null || t[d].disabled || (e = t[d]);
    }
    e !== null && (e.selected = !0);
  }
}
function il(t, e) {
  if (e.dangerouslySetInnerHTML != null) throw Error(pt(91));
  return ne({}, e, { value: void 0, defaultValue: void 0, children: "" + t._wrapperState.initialValue });
}
function ju(t, e) {
  var n = e.value;
  if (n == null) {
    if (n = e.children, e = e.defaultValue, n != null) {
      if (e != null) throw Error(pt(92));
      if (Lr(n)) {
        if (1 < n.length) throw Error(pt(93));
        n = n[0];
      }
      e = n;
    }
    e == null && (e = ""), n = e;
  }
  t._wrapperState = { initialValue: si(n) };
}
function Gf(t, e) {
  var n = si(e.value), o = si(e.defaultValue);
  n != null && (n = "" + n, n !== t.value && (t.value = n), e.defaultValue == null && t.defaultValue !== n && (t.defaultValue = n)), o != null && (t.defaultValue = "" + o);
}
function Lu(t) {
  var e = t.textContent;
  e === t._wrapperState.initialValue && e !== "" && e !== null && (t.value = e);
}
function qf(t) {
  switch (t) {
    case "svg":
      return "http://www.w3.org/2000/svg";
    case "math":
      return "http://www.w3.org/1998/Math/MathML";
    default:
      return "http://www.w3.org/1999/xhtml";
  }
}
function rl(t, e) {
  return t == null || t === "http://www.w3.org/1999/xhtml" ? qf(e) : t === "http://www.w3.org/2000/svg" && e === "foreignObject" ? "http://www.w3.org/1999/xhtml" : t;
}
var Oa, Zf = function (t) {
  return typeof MSApp < "u" && MSApp.execUnsafeLocalFunction ? function (e, n, o, d) {
    MSApp.execUnsafeLocalFunction(function () {
      return t(e, n, o, d);
    });
  } : t;
}(function (t, e) {
  if (t.namespaceURI !== "http://www.w3.org/2000/svg" || "innerHTML" in t) t.innerHTML = e;
  else {
    for (Oa = Oa || document.createElement("div"), Oa.innerHTML = "<svg>" + e.valueOf().toString() + "</svg>", e = Oa.firstChild; t.firstChild;) t.removeChild(t.firstChild);
    for (; e.firstChild;) t.appendChild(e.firstChild);
  }
});
function qr(t, e) {
  if (e) {
    var n = t.firstChild;
    if (n && n === t.lastChild && n.nodeType === 3) {
      n.nodeValue = e;
      return;
    }
  }
  t.textContent = e;
}
var Rr = {
  animationIterationCount: !0,
  aspectRatio: !0,
  borderImageOutset: !0,
  borderImageSlice: !0,
  borderImageWidth: !0,
  boxFlex: !0,
  boxFlexGroup: !0,
  boxOrdinalGroup: !0,
  columnCount: !0,
  columns: !0,
  flex: !0,
  flexGrow: !0,
  flexPositive: !0,
  flexShrink: !0,
  flexNegative: !0,
  flexOrder: !0,
  gridArea: !0,
  gridRow: !0,
  gridRowEnd: !0,
  gridRowSpan: !0,
  gridRowStart: !0,
  gridColumn: !0,
  gridColumnEnd: !0,
  gridColumnSpan: !0,
  gridColumnStart: !0,
  fontWeight: !0,
  lineClamp: !0,
  lineHeight: !0,
  opacity: !0,
  order: !0,
  orphans: !0,
  tabSize: !0,
  widows: !0,
  zIndex: !0,
  zoom: !0,
  fillOpacity: !0,
  floodOpacity: !0,
  stopOpacity: !0,
  strokeDasharray: !0,
  strokeDashoffset: !0,
  strokeMiterlimit: !0,
  strokeOpacity: !0,
  strokeWidth: !0
}, Mm = ["Webkit", "ms", "Moz", "O"];
Object.keys(Rr).forEach(function (t) {
  Mm.forEach(function (e) {
    e = e + t.charAt(0).toUpperCase() + t.substring(1), Rr[e] = Rr[t];
  });
});
function Qf(t, e, n) {
  return e == null || typeof e == "boolean" || e === "" ? "" : n || typeof e != "number" || e === 0 || Rr.hasOwnProperty(t) && Rr[t] ? ("" + e).trim() : e + "px";
}
function Kf(t, e) {
  t = t.style;
  for (var n in e) if (e.hasOwnProperty(n)) {
    var o = n.indexOf("--") === 0, d = Qf(n, e[n], o);
    n === "float" && (n = "cssFloat"), o ? t.setProperty(n, d) : t[n] = d;
  }
}
var Am = ne({ menuitem: !0 }, { area: !0, base: !0, br: !0, col: !0, embed: !0, hr: !0, img: !0, input: !0, keygen: !0, link: !0, meta: !0, param: !0, source: !0, track: !0, wbr: !0 });
function al(t, e) {
  if (e) {
    if (Am[t] && (e.children != null || e.dangerouslySetInnerHTML != null)) throw Error(pt(137, t));
    if (e.dangerouslySetInnerHTML != null) {
      if (e.children != null) throw Error(pt(60));
      if (typeof e.dangerouslySetInnerHTML != "object" || !("__html" in e.dangerouslySetInnerHTML)) throw Error(pt(61));
    }
    if (e.style != null && typeof e.style != "object") throw Error(pt(62));
  }
}
function ol(t, e) {
  if (t.indexOf("-") === -1) return typeof e.is == "string";
  switch (t) {
    case "annotation-xml":
    case "color-profile":
    case "font-face":
    case "font-face-src":
    case "font-face-uri":
    case "font-face-format":
    case "font-face-name":
    case "missing-glyph":
      return !1;
    default:
      return !0;
  }
}
var sl = null;
function ic(t) {
  return t = t.target || t.srcElement || window, t.correspondingUseElement && (t = t.correspondingUseElement), t.nodeType === 3 ? t.parentNode : t;
}
var ll = null, $i = null, tr = null;
function Fu(t) {
  if (t = pa(t)) {
    if (typeof ll != "function") throw Error(pt(280));
    var e = t.stateNode;
    e && (e = Xo(e), ll(t.stateNode, t.type, e));
  }
}
function Jf(t) {
  $i ? tr ? tr.push(t) : tr = [t] : $i = t;
}
function $f() {
  if ($i) {
    var t = $i, e = tr;
    if (tr = $i = null, Fu(t), e) for (t = 0; t < e.length; t++) Fu(e[t]);
  }
}
function td(t, e) {
  return t(e);
}
function ed() {
}
var gs = !1;
function nd(t, e, n) {
  if (gs) return t(e, n);
  gs = !0;
  try {
    return td(t, e, n);
  } finally {
    gs = !1, ($i !== null || tr !== null) && (ed(), $f());
  }
}
function Zr(t, e) {
  var n = t.stateNode;
  if (n === null) return null;
  var o = Xo(n);
  if (o === null) return null;
  n = o[e];
  t: switch (e) {
    case "onClick":
    case "onClickCapture":
    case "onDoubleClick":
    case "onDoubleClickCapture":
    case "onMouseDown":
    case "onMouseDownCapture":
    case "onMouseMove":
    case "onMouseMoveCapture":
    case "onMouseUp":
    case "onMouseUpCapture":
    case "onMouseEnter":
      (o = !o.disabled) || (t = t.type, o = !(t === "button" || t === "input" || t === "select" || t === "textarea")), t = !o;
      break t;
    default:
      t = !1;
  }
  if (t) return null;
  if (n && typeof n != "function") throw Error(pt(231, e, typeof n));
  return n;
}
var cl = !1;
if (Nn) try {
  var kr = {};
  Object.defineProperty(kr, "passive", {
    get: function () {
      cl = !0;
    }
  }), window.addEventListener("test", kr, kr), window.removeEventListener("test", kr, kr);
} catch {
  cl = !1;
}
function Dm(t, e, n, o, d, g, b, _, w) {
  var E = Array.prototype.slice.call(arguments, 3);
  try {
    e.apply(n, E);
  } catch (j) {
    this.onError(j);
  }
}
var Br = !1, co = null, uo = !1, ul = null, jm = {
  onError: function (t) {
    Br = !0, co = t;
  }
};
function Lm(t, e, n, o, d, g, b, _, w) {
  Br = !1, co = null, Dm.apply(jm, arguments);
}
function Fm(t, e, n, o, d, g, b, _, w) {
  if (Lm.apply(this, arguments), Br) {
    if (Br) {
      var E = co;
      Br = !1, co = null;
    } else throw Error(pt(198));
    uo || (uo = !0, ul = E);
  }
}
function Di(t) {
  var e = t, n = t;
  if (t.alternate) for (; e.return;) e = e.return;
  else {
    t = e;
    do
      e = t, e.flags & 4098 && (n = e.return), t = e.return;
    while (t);
  }
  return e.tag === 3 ? n : null;
}
function id(t) {
  if (t.tag === 13) {
    var e = t.memoizedState;
    if (e === null && (t = t.alternate, t !== null && (e = t.memoizedState)), e !== null) return e.dehydrated;
  }
  return null;
}
function Iu(t) {
  if (Di(t) !== t) throw Error(pt(188));
}
function Im(t) {
  var e = t.alternate;
  if (!e) {
    if (e = Di(t), e === null) throw Error(pt(188));
    return e !== t ? null : t;
  }
  for (var n = t, o = e; ;) {
    var d = n.return;
    if (d === null) break;
    var g = d.alternate;
    if (g === null) {
      if (o = d.return, o !== null) {
        n = o;
        continue;
      }
      break;
    }
    if (d.child === g.child) {
      for (g = d.child; g;) {
        if (g === n) return Iu(d), t;
        if (g === o) return Iu(d), e;
        g = g.sibling;
      }
      throw Error(pt(188));
    }
    if (n.return !== o.return) n = d, o = g;
    else {
      for (var b = !1, _ = d.child; _;) {
        if (_ === n) {
          b = !0, n = d, o = g;
          break;
        }
        if (_ === o) {
          b = !0, o = d, n = g;
          break;
        }
        _ = _.sibling;
      }
      if (!b) {
        for (_ = g.child; _;) {
          if (_ === n) {
            b = !0, n = g, o = d;
            break;
          }
          if (_ === o) {
            b = !0, o = g, n = d;
            break;
          }
          _ = _.sibling;
        }
        if (!b) throw Error(pt(189));
      }
    }
    if (n.alternate !== o) throw Error(pt(190));
  }
  if (n.tag !== 3) throw Error(pt(188));
  return n.stateNode.current === n ? t : e;
}
function rd(t) {
  return t = Im(t), t !== null ? ad(t) : null;
}
function ad(t) {
  if (t.tag === 5 || t.tag === 6) return t;
  for (t = t.child; t !== null;) {
    var e = ad(t);
    if (e !== null) return e;
    t = t.sibling;
  }
  return null;
}
var od = We.unstable_scheduleCallback, Ru = We.unstable_cancelCallback, Rm = We.unstable_shouldYield, Bm = We.unstable_requestPaint, ae = We.unstable_now, zm = We.unstable_getCurrentPriorityLevel, rc = We.unstable_ImmediatePriority, sd = We.unstable_UserBlockingPriority, ho = We.unstable_NormalPriority, Nm = We.unstable_LowPriority, ld = We.unstable_IdlePriority, No = null, Mn = null;
function Um(t) {
  if (Mn && typeof Mn.onCommitFiberRoot == "function") try {
    Mn.onCommitFiberRoot(No, t, void 0, (t.current.flags & 128) === 128);
  } catch {
  }
}
var mn = Math.clz32 ? Math.clz32 : Xm, Hm = Math.log, Wm = Math.LN2;
function Xm(t) {
  return t >>>= 0, t === 0 ? 32 : 31 - (Hm(t) / Wm | 0) | 0;
}
var Pa = 64, Ma = 4194304;
function Fr(t) {
  switch (t & -t) {
    case 1:
      return 1;
    case 2:
      return 2;
    case 4:
      return 4;
    case 8:
      return 8;
    case 16:
      return 16;
    case 32:
      return 32;
    case 64:
    case 128:
    case 256:
    case 512:
    case 1024:
    case 2048:
    case 4096:
    case 8192:
    case 16384:
    case 32768:
    case 65536:
    case 131072:
    case 262144:
    case 524288:
    case 1048576:
    case 2097152:
      return t & 4194240;
    case 4194304:
    case 8388608:
    case 16777216:
    case 33554432:
    case 67108864:
      return t & 130023424;
    case 134217728:
      return 134217728;
    case 268435456:
      return 268435456;
    case 536870912:
      return 536870912;
    case 1073741824:
      return 1073741824;
    default:
      return t;
  }
}
function fo(t, e) {
  var n = t.pendingLanes;
  if (n === 0) return 0;
  var o = 0, d = t.suspendedLanes, g = t.pingedLanes, b = n & 268435455;
  if (b !== 0) {
    var _ = b & ~d;
    _ !== 0 ? o = Fr(_) : (g &= b, g !== 0 && (o = Fr(g)));
  } else b = n & ~d, b !== 0 ? o = Fr(b) : g !== 0 && (o = Fr(g));
  if (o === 0) return 0;
  if (e !== 0 && e !== o && !(e & d) && (d = o & -o, g = e & -e, d >= g || d === 16 && (g & 4194240) !== 0)) return e;
  if (o & 4 && (o |= n & 16), e = t.entangledLanes, e !== 0) for (t = t.entanglements, e &= o; 0 < e;) n = 31 - mn(e), d = 1 << n, o |= t[n], e &= ~d;
  return o;
}
function Ym(t, e) {
  switch (t) {
    case 1:
    case 2:
    case 4:
      return e + 250;
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
    case 256:
    case 512:
    case 1024:
    case 2048:
    case 4096:
    case 8192:
    case 16384:
    case 32768:
    case 65536:
    case 131072:
    case 262144:
    case 524288:
    case 1048576:
    case 2097152:
      return e + 5e3;
    case 4194304:
    case 8388608:
    case 16777216:
    case 33554432:
    case 67108864:
      return -1;
    case 134217728:
    case 268435456:
    case 536870912:
    case 1073741824:
      return -1;
    default:
      return -1;
  }
}
function Vm(t, e) {
  for (var n = t.suspendedLanes, o = t.pingedLanes, d = t.expirationTimes, g = t.pendingLanes; 0 < g;) {
    var b = 31 - mn(g), _ = 1 << b, w = d[b];
    w === -1 ? (!(_ & n) || _ & o) && (d[b] = Ym(_, e)) : w <= e && (t.expiredLanes |= _), g &= ~_;
  }
}
function hl(t) {
  return t = t.pendingLanes & -1073741825, t !== 0 ? t : t & 1073741824 ? 1073741824 : 0;
}
function cd() {
  var t = Pa;
  return Pa <<= 1, !(Pa & 4194240) && (Pa = 64), t;
}
function ms(t) {
  for (var e = [], n = 0; 31 > n; n++) e.push(t);
  return e;
}
function fa(t, e, n) {
  t.pendingLanes |= e, e !== 536870912 && (t.suspendedLanes = 0, t.pingedLanes = 0), t = t.eventTimes, e = 31 - mn(e), t[e] = n;
}
function Gm(t, e) {
  var n = t.pendingLanes & ~e;
  t.pendingLanes = e, t.suspendedLanes = 0, t.pingedLanes = 0, t.expiredLanes &= e, t.mutableReadLanes &= e, t.entangledLanes &= e, e = t.entanglements;
  var o = t.eventTimes;
  for (t = t.expirationTimes; 0 < n;) {
    var d = 31 - mn(n), g = 1 << d;
    e[d] = 0, o[d] = -1, t[d] = -1, n &= ~g;
  }
}
function ac(t, e) {
  var n = t.entangledLanes |= e;
  for (t = t.entanglements; n;) {
    var o = 31 - mn(n), d = 1 << o;
    d & e | t[o] & e && (t[o] |= e), n &= ~d;
  }
}
var Xt = 0;
function ud(t) {
  return t &= -t, 1 < t ? 4 < t ? t & 268435455 ? 16 : 536870912 : 4 : 1;
}
var hd, oc, fd, dd, pd, fl = !1, Aa = [], $n = null, ti = null, ei = null, Qr = /* @__PURE__ */ new Map(), Kr = /* @__PURE__ */ new Map(), qn = [], qm = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");
function Bu(t, e) {
  switch (t) {
    case "focusin":
    case "focusout":
      $n = null;
      break;
    case "dragenter":
    case "dragleave":
      ti = null;
      break;
    case "mouseover":
    case "mouseout":
      ei = null;
      break;
    case "pointerover":
    case "pointerout":
      Qr.delete(e.pointerId);
      break;
    case "gotpointercapture":
    case "lostpointercapture":
      Kr.delete(e.pointerId);
  }
}
function Tr(t, e, n, o, d, g) {
  return t === null || t.nativeEvent !== g ? (t = { blockedOn: e, domEventName: n, eventSystemFlags: o, nativeEvent: g, targetContainers: [d] }, e !== null && (e = pa(e), e !== null && oc(e)), t) : (t.eventSystemFlags |= o, e = t.targetContainers, d !== null && e.indexOf(d) === -1 && e.push(d), t);
}
function Zm(t, e, n, o, d) {
  switch (e) {
    case "focusin":
      return $n = Tr($n, t, e, n, o, d), !0;
    case "dragenter":
      return ti = Tr(ti, t, e, n, o, d), !0;
    case "mouseover":
      return ei = Tr(ei, t, e, n, o, d), !0;
    case "pointerover":
      var g = d.pointerId;
      return Qr.set(g, Tr(Qr.get(g) || null, t, e, n, o, d)), !0;
    case "gotpointercapture":
      return g = d.pointerId, Kr.set(g, Tr(Kr.get(g) || null, t, e, n, o, d)), !0;
  }
  return !1;
}
function gd(t) {
  var e = _i(t.target);
  if (e !== null) {
    var n = Di(e);
    if (n !== null) {
      if (e = n.tag, e === 13) {
        if (e = id(n), e !== null) {
          t.blockedOn = e, pd(t.priority, function () {
            fd(n);
          });
          return;
        }
      } else if (e === 3 && n.stateNode.current.memoizedState.isDehydrated) {
        t.blockedOn = n.tag === 3 ? n.stateNode.containerInfo : null;
        return;
      }
    }
  }
  t.blockedOn = null;
}
function Ka(t) {
  if (t.blockedOn !== null) return !1;
  for (var e = t.targetContainers; 0 < e.length;) {
    var n = dl(t.domEventName, t.eventSystemFlags, e[0], t.nativeEvent);
    if (n === null) {
      n = t.nativeEvent;
      var o = new n.constructor(n.type, n);
      sl = o, n.target.dispatchEvent(o), sl = null;
    } else return e = pa(n), e !== null && oc(e), t.blockedOn = n, !1;
    e.shift();
  }
  return !0;
}
function zu(t, e, n) {
  Ka(t) && n.delete(e);
}
function Qm() {
  fl = !1, $n !== null && Ka($n) && ($n = null), ti !== null && Ka(ti) && (ti = null), ei !== null && Ka(ei) && (ei = null), Qr.forEach(zu), Kr.forEach(zu);
}
function Or(t, e) {
  t.blockedOn === e && (t.blockedOn = null, fl || (fl = !0, We.unstable_scheduleCallback(We.unstable_NormalPriority, Qm)));
}
function Jr(t) {
  function e(d) {
    return Or(d, t);
  }
  if (0 < Aa.length) {
    Or(Aa[0], t);
    for (var n = 1; n < Aa.length; n++) {
      var o = Aa[n];
      o.blockedOn === t && (o.blockedOn = null);
    }
  }
  for ($n !== null && Or($n, t), ti !== null && Or(ti, t), ei !== null && Or(ei, t), Qr.forEach(e), Kr.forEach(e), n = 0; n < qn.length; n++) o = qn[n], o.blockedOn === t && (o.blockedOn = null);
  for (; 0 < qn.length && (n = qn[0], n.blockedOn === null);) gd(n), n.blockedOn === null && qn.shift();
}
var er = Xn.ReactCurrentBatchConfig, po = !0;
function Km(t, e, n, o) {
  var d = Xt, g = er.transition;
  er.transition = null;
  try {
    Xt = 1, sc(t, e, n, o);
  } finally {
    Xt = d, er.transition = g;
  }
}
function Jm(t, e, n, o) {
  var d = Xt, g = er.transition;
  er.transition = null;
  try {
    Xt = 4, sc(t, e, n, o);
  } finally {
    Xt = d, er.transition = g;
  }
}
function sc(t, e, n, o) {
  if (po) {
    var d = dl(t, e, n, o);
    if (d === null) ks(t, e, o, go, n), Bu(t, o);
    else if (Zm(d, t, e, n, o)) o.stopPropagation();
    else if (Bu(t, o), e & 4 && -1 < qm.indexOf(t)) {
      for (; d !== null;) {
        var g = pa(d);
        if (g !== null && hd(g), g = dl(t, e, n, o), g === null && ks(t, e, o, go, n), g === d) break;
        d = g;
      }
      d !== null && o.stopPropagation();
    } else ks(t, e, o, null, n);
  }
}
var go = null;
function dl(t, e, n, o) {
  if (go = null, t = ic(o), t = _i(t), t !== null) if (e = Di(t), e === null) t = null;
  else if (n = e.tag, n === 13) {
    if (t = id(e), t !== null) return t;
    t = null;
  } else if (n === 3) {
    if (e.stateNode.current.memoizedState.isDehydrated) return e.tag === 3 ? e.stateNode.containerInfo : null;
    t = null;
  } else e !== t && (t = null);
  return go = t, null;
}
function md(t) {
  switch (t) {
    case "cancel":
    case "click":
    case "close":
    case "contextmenu":
    case "copy":
    case "cut":
    case "auxclick":
    case "dblclick":
    case "dragend":
    case "dragstart":
    case "drop":
    case "focusin":
    case "focusout":
    case "input":
    case "invalid":
    case "keydown":
    case "keypress":
    case "keyup":
    case "mousedown":
    case "mouseup":
    case "paste":
    case "pause":
    case "play":
    case "pointercancel":
    case "pointerdown":
    case "pointerup":
    case "ratechange":
    case "reset":
    case "resize":
    case "seeked":
    case "submit":
    case "touchcancel":
    case "touchend":
    case "touchstart":
    case "volumechange":
    case "change":
    case "selectionchange":
    case "textInput":
    case "compositionstart":
    case "compositionend":
    case "compositionupdate":
    case "beforeblur":
    case "afterblur":
    case "beforeinput":
    case "blur":
    case "fullscreenchange":
    case "focus":
    case "hashchange":
    case "popstate":
    case "select":
    case "selectstart":
      return 1;
    case "drag":
    case "dragenter":
    case "dragexit":
    case "dragleave":
    case "dragover":
    case "mousemove":
    case "mouseout":
    case "mouseover":
    case "pointermove":
    case "pointerout":
    case "pointerover":
    case "scroll":
    case "toggle":
    case "touchmove":
    case "wheel":
    case "mouseenter":
    case "mouseleave":
    case "pointerenter":
    case "pointerleave":
      return 4;
    case "message":
      switch (zm()) {
        case rc:
          return 1;
        case sd:
          return 4;
        case ho:
        case Nm:
          return 16;
        case ld:
          return 536870912;
        default:
          return 16;
      }
    default:
      return 16;
  }
}
var Kn = null, lc = null, Ja = null;
function vd() {
  if (Ja) return Ja;
  var t, e = lc, n = e.length, o, d = "value" in Kn ? Kn.value : Kn.textContent, g = d.length;
  for (t = 0; t < n && e[t] === d[t]; t++);
  var b = n - t;
  for (o = 1; o <= b && e[n - o] === d[g - o]; o++);
  return Ja = d.slice(t, 1 < o ? 1 - o : void 0);
}
function $a(t) {
  var e = t.keyCode;
  return "charCode" in t ? (t = t.charCode, t === 0 && e === 13 && (t = 13)) : t = e, t === 10 && (t = 13), 32 <= t || t === 13 ? t : 0;
}
function Da() {
  return !0;
}
function Nu() {
  return !1;
}
function Ye(t) {
  function e(n, o, d, g, b) {
    this._reactName = n, this._targetInst = d, this.type = o, this.nativeEvent = g, this.target = b, this.currentTarget = null;
    for (var _ in t) t.hasOwnProperty(_) && (n = t[_], this[_] = n ? n(g) : g[_]);
    return this.isDefaultPrevented = (g.defaultPrevented != null ? g.defaultPrevented : g.returnValue === !1) ? Da : Nu, this.isPropagationStopped = Nu, this;
  }
  return ne(e.prototype, {
    preventDefault: function () {
      this.defaultPrevented = !0;
      var n = this.nativeEvent;
      n && (n.preventDefault ? n.preventDefault() : typeof n.returnValue != "unknown" && (n.returnValue = !1), this.isDefaultPrevented = Da);
    }, stopPropagation: function () {
      var n = this.nativeEvent;
      n && (n.stopPropagation ? n.stopPropagation() : typeof n.cancelBubble != "unknown" && (n.cancelBubble = !0), this.isPropagationStopped = Da);
    }, persist: function () {
    }, isPersistent: Da
  }), e;
}
var pr = {
  eventPhase: 0, bubbles: 0, cancelable: 0, timeStamp: function (t) {
    return t.timeStamp || Date.now();
  }, defaultPrevented: 0, isTrusted: 0
}, cc = Ye(pr), da = ne({}, pr, { view: 0, detail: 0 }), $m = Ye(da), vs, ys, Pr, Uo = ne({}, da, {
  screenX: 0, screenY: 0, clientX: 0, clientY: 0, pageX: 0, pageY: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, getModifierState: uc, button: 0, buttons: 0, relatedTarget: function (t) {
    return t.relatedTarget === void 0 ? t.fromElement === t.srcElement ? t.toElement : t.fromElement : t.relatedTarget;
  }, movementX: function (t) {
    return "movementX" in t ? t.movementX : (t !== Pr && (Pr && t.type === "mousemove" ? (vs = t.screenX - Pr.screenX, ys = t.screenY - Pr.screenY) : ys = vs = 0, Pr = t), vs);
  }, movementY: function (t) {
    return "movementY" in t ? t.movementY : ys;
  }
}), Uu = Ye(Uo), t1 = ne({}, Uo, { dataTransfer: 0 }), e1 = Ye(t1), n1 = ne({}, da, { relatedTarget: 0 }), bs = Ye(n1), i1 = ne({}, pr, { animationName: 0, elapsedTime: 0, pseudoElement: 0 }), r1 = Ye(i1), a1 = ne({}, pr, {
  clipboardData: function (t) {
    return "clipboardData" in t ? t.clipboardData : window.clipboardData;
  }
}), o1 = Ye(a1), s1 = ne({}, pr, { data: 0 }), Hu = Ye(s1), l1 = {
  Esc: "Escape",
  Spacebar: " ",
  Left: "ArrowLeft",
  Up: "ArrowUp",
  Right: "ArrowRight",
  Down: "ArrowDown",
  Del: "Delete",
  Win: "OS",
  Menu: "ContextMenu",
  Apps: "ContextMenu",
  Scroll: "ScrollLock",
  MozPrintableKey: "Unidentified"
}, c1 = {
  8: "Backspace",
  9: "Tab",
  12: "Clear",
  13: "Enter",
  16: "Shift",
  17: "Control",
  18: "Alt",
  19: "Pause",
  20: "CapsLock",
  27: "Escape",
  32: " ",
  33: "PageUp",
  34: "PageDown",
  35: "End",
  36: "Home",
  37: "ArrowLeft",
  38: "ArrowUp",
  39: "ArrowRight",
  40: "ArrowDown",
  45: "Insert",
  46: "Delete",
  112: "F1",
  113: "F2",
  114: "F3",
  115: "F4",
  116: "F5",
  117: "F6",
  118: "F7",
  119: "F8",
  120: "F9",
  121: "F10",
  122: "F11",
  123: "F12",
  144: "NumLock",
  145: "ScrollLock",
  224: "Meta"
}, u1 = { Alt: "altKey", Control: "ctrlKey", Meta: "metaKey", Shift: "shiftKey" };
function h1(t) {
  var e = this.nativeEvent;
  return e.getModifierState ? e.getModifierState(t) : (t = u1[t]) ? !!e[t] : !1;
}
function uc() {
  return h1;
}
var f1 = ne({}, da, {
  key: function (t) {
    if (t.key) {
      var e = l1[t.key] || t.key;
      if (e !== "Unidentified") return e;
    }
    return t.type === "keypress" ? (t = $a(t), t === 13 ? "Enter" : String.fromCharCode(t)) : t.type === "keydown" || t.type === "keyup" ? c1[t.keyCode] || "Unidentified" : "";
  }, code: 0, location: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, repeat: 0, locale: 0, getModifierState: uc, charCode: function (t) {
    return t.type === "keypress" ? $a(t) : 0;
  }, keyCode: function (t) {
    return t.type === "keydown" || t.type === "keyup" ? t.keyCode : 0;
  }, which: function (t) {
    return t.type === "keypress" ? $a(t) : t.type === "keydown" || t.type === "keyup" ? t.keyCode : 0;
  }
}), d1 = Ye(f1), p1 = ne({}, Uo, { pointerId: 0, width: 0, height: 0, pressure: 0, tangentialPressure: 0, tiltX: 0, tiltY: 0, twist: 0, pointerType: 0, isPrimary: 0 }), Wu = Ye(p1), g1 = ne({}, da, { touches: 0, targetTouches: 0, changedTouches: 0, altKey: 0, metaKey: 0, ctrlKey: 0, shiftKey: 0, getModifierState: uc }), m1 = Ye(g1), v1 = ne({}, pr, { propertyName: 0, elapsedTime: 0, pseudoElement: 0 }), y1 = Ye(v1), b1 = ne({}, Uo, {
  deltaX: function (t) {
    return "deltaX" in t ? t.deltaX : "wheelDeltaX" in t ? -t.wheelDeltaX : 0;
  },
  deltaY: function (t) {
    return "deltaY" in t ? t.deltaY : "wheelDeltaY" in t ? -t.wheelDeltaY : "wheelDelta" in t ? -t.wheelDelta : 0;
  },
  deltaZ: 0,
  deltaMode: 0
}), x1 = Ye(b1), _1 = [9, 13, 27, 32], hc = Nn && "CompositionEvent" in window, zr = null;
Nn && "documentMode" in document && (zr = document.documentMode);
var w1 = Nn && "TextEvent" in window && !zr, yd = Nn && (!hc || zr && 8 < zr && 11 >= zr), Xu = " ", Yu = !1;
function bd(t, e) {
  switch (t) {
    case "keyup":
      return _1.indexOf(e.keyCode) !== -1;
    case "keydown":
      return e.keyCode !== 229;
    case "keypress":
    case "mousedown":
    case "focusout":
      return !0;
    default:
      return !1;
  }
}
function xd(t) {
  return t = t.detail, typeof t == "object" && "data" in t ? t.data : null;
}
var Ui = !1;
function C1(t, e) {
  switch (t) {
    case "compositionend":
      return xd(e);
    case "keypress":
      return e.which !== 32 ? null : (Yu = !0, Xu);
    case "textInput":
      return t = e.data, t === Xu && Yu ? null : t;
    default:
      return null;
  }
}
function S1(t, e) {
  if (Ui) return t === "compositionend" || !hc && bd(t, e) ? (t = vd(), Ja = lc = Kn = null, Ui = !1, t) : null;
  switch (t) {
    case "paste":
      return null;
    case "keypress":
      if (!(e.ctrlKey || e.altKey || e.metaKey) || e.ctrlKey && e.altKey) {
        if (e.char && 1 < e.char.length) return e.char;
        if (e.which) return String.fromCharCode(e.which);
      }
      return null;
    case "compositionend":
      return yd && e.locale !== "ko" ? null : e.data;
    default:
      return null;
  }
}
var E1 = { color: !0, date: !0, datetime: !0, "datetime-local": !0, email: !0, month: !0, number: !0, password: !0, range: !0, search: !0, tel: !0, text: !0, time: !0, url: !0, week: !0 };
function Vu(t) {
  var e = t && t.nodeName && t.nodeName.toLowerCase();
  return e === "input" ? !!E1[t.type] : e === "textarea";
}
function _d(t, e, n, o) {
  Jf(o), e = mo(e, "onChange"), 0 < e.length && (n = new cc("onChange", "change", null, n, o), t.push({ event: n, listeners: e }));
}
var Nr = null, $r = null;
function k1(t) {
  Dd(t, 0);
}
function Ho(t) {
  var e = Xi(t);
  if (Yf(e)) return t;
}
function T1(t, e) {
  if (t === "change") return e;
}
var wd = !1;
if (Nn) {
  var xs;
  if (Nn) {
    var _s = "oninput" in document;
    if (!_s) {
      var Gu = document.createElement("div");
      Gu.setAttribute("oninput", "return;"), _s = typeof Gu.oninput == "function";
    }
    xs = _s;
  } else xs = !1;
  wd = xs && (!document.documentMode || 9 < document.documentMode);
}
function qu() {
  Nr && (Nr.detachEvent("onpropertychange", Cd), $r = Nr = null);
}
function Cd(t) {
  if (t.propertyName === "value" && Ho($r)) {
    var e = [];
    _d(e, $r, t, ic(t)), nd(k1, e);
  }
}
function O1(t, e, n) {
  t === "focusin" ? (qu(), Nr = e, $r = n, Nr.attachEvent("onpropertychange", Cd)) : t === "focusout" && qu();
}
function P1(t) {
  if (t === "selectionchange" || t === "keyup" || t === "keydown") return Ho($r);
}
function M1(t, e) {
  if (t === "click") return Ho(e);
}
function A1(t, e) {
  if (t === "input" || t === "change") return Ho(e);
}
function D1(t, e) {
  return t === e && (t !== 0 || 1 / t === 1 / e) || t !== t && e !== e;
}
var yn = typeof Object.is == "function" ? Object.is : D1;
function ta(t, e) {
  if (yn(t, e)) return !0;
  if (typeof t != "object" || t === null || typeof e != "object" || e === null) return !1;
  var n = Object.keys(t), o = Object.keys(e);
  if (n.length !== o.length) return !1;
  for (o = 0; o < n.length; o++) {
    var d = n[o];
    if (!Zs.call(e, d) || !yn(t[d], e[d])) return !1;
  }
  return !0;
}
function Zu(t) {
  for (; t && t.firstChild;) t = t.firstChild;
  return t;
}
function Qu(t, e) {
  var n = Zu(t);
  t = 0;
  for (var o; n;) {
    if (n.nodeType === 3) {
      if (o = t + n.textContent.length, t <= e && o >= e) return { node: n, offset: e - t };
      t = o;
    }
    t: {
      for (; n;) {
        if (n.nextSibling) {
          n = n.nextSibling;
          break t;
        }
        n = n.parentNode;
      }
      n = void 0;
    }
    n = Zu(n);
  }
}
function Sd(t, e) {
  return t && e ? t === e ? !0 : t && t.nodeType === 3 ? !1 : e && e.nodeType === 3 ? Sd(t, e.parentNode) : "contains" in t ? t.contains(e) : t.compareDocumentPosition ? !!(t.compareDocumentPosition(e) & 16) : !1 : !1;
}
function Ed() {
  for (var t = window, e = lo(); e instanceof t.HTMLIFrameElement;) {
    try {
      var n = typeof e.contentWindow.location.href == "string";
    } catch {
      n = !1;
    }
    if (n) t = e.contentWindow;
    else break;
    e = lo(t.document);
  }
  return e;
}
function fc(t) {
  var e = t && t.nodeName && t.nodeName.toLowerCase();
  return e && (e === "input" && (t.type === "text" || t.type === "search" || t.type === "tel" || t.type === "url" || t.type === "password") || e === "textarea" || t.contentEditable === "true");
}
function j1(t) {
  var e = Ed(), n = t.focusedElem, o = t.selectionRange;
  if (e !== n && n && n.ownerDocument && Sd(n.ownerDocument.documentElement, n)) {
    if (o !== null && fc(n)) {
      if (e = o.start, t = o.end, t === void 0 && (t = e), "selectionStart" in n) n.selectionStart = e, n.selectionEnd = Math.min(t, n.value.length);
      else if (t = (e = n.ownerDocument || document) && e.defaultView || window, t.getSelection) {
        t = t.getSelection();
        var d = n.textContent.length, g = Math.min(o.start, d);
        o = o.end === void 0 ? g : Math.min(o.end, d), !t.extend && g > o && (d = o, o = g, g = d), d = Qu(n, g);
        var b = Qu(
          n,
          o
        );
        d && b && (t.rangeCount !== 1 || t.anchorNode !== d.node || t.anchorOffset !== d.offset || t.focusNode !== b.node || t.focusOffset !== b.offset) && (e = e.createRange(), e.setStart(d.node, d.offset), t.removeAllRanges(), g > o ? (t.addRange(e), t.extend(b.node, b.offset)) : (e.setEnd(b.node, b.offset), t.addRange(e)));
      }
    }
    for (e = [], t = n; t = t.parentNode;) t.nodeType === 1 && e.push({ element: t, left: t.scrollLeft, top: t.scrollTop });
    for (typeof n.focus == "function" && n.focus(), n = 0; n < e.length; n++) t = e[n], t.element.scrollLeft = t.left, t.element.scrollTop = t.top;
  }
}
var L1 = Nn && "documentMode" in document && 11 >= document.documentMode, Hi = null, pl = null, Ur = null, gl = !1;
function Ku(t, e, n) {
  var o = n.window === n ? n.document : n.nodeType === 9 ? n : n.ownerDocument;
  gl || Hi == null || Hi !== lo(o) || (o = Hi, "selectionStart" in o && fc(o) ? o = { start: o.selectionStart, end: o.selectionEnd } : (o = (o.ownerDocument && o.ownerDocument.defaultView || window).getSelection(), o = { anchorNode: o.anchorNode, anchorOffset: o.anchorOffset, focusNode: o.focusNode, focusOffset: o.focusOffset }), Ur && ta(Ur, o) || (Ur = o, o = mo(pl, "onSelect"), 0 < o.length && (e = new cc("onSelect", "select", null, e, n), t.push({ event: e, listeners: o }), e.target = Hi)));
}
function ja(t, e) {
  var n = {};
  return n[t.toLowerCase()] = e.toLowerCase(), n["Webkit" + t] = "webkit" + e, n["Moz" + t] = "moz" + e, n;
}
var Wi = { animationend: ja("Animation", "AnimationEnd"), animationiteration: ja("Animation", "AnimationIteration"), animationstart: ja("Animation", "AnimationStart"), transitionend: ja("Transition", "TransitionEnd") }, ws = {}, kd = {};
Nn && (kd = document.createElement("div").style, "AnimationEvent" in window || (delete Wi.animationend.animation, delete Wi.animationiteration.animation, delete Wi.animationstart.animation), "TransitionEvent" in window || delete Wi.transitionend.transition);
function Wo(t) {
  if (ws[t]) return ws[t];
  if (!Wi[t]) return t;
  var e = Wi[t], n;
  for (n in e) if (e.hasOwnProperty(n) && n in kd) return ws[t] = e[n];
  return t;
}
var Td = Wo("animationend"), Od = Wo("animationiteration"), Pd = Wo("animationstart"), Md = Wo("transitionend"), Ad = /* @__PURE__ */ new Map(), Ju = "abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");
function ci(t, e) {
  Ad.set(t, e), Ai(e, [t]);
}
for (var Cs = 0; Cs < Ju.length; Cs++) {
  var Ss = Ju[Cs], F1 = Ss.toLowerCase(), I1 = Ss[0].toUpperCase() + Ss.slice(1);
  ci(F1, "on" + I1);
}
ci(Td, "onAnimationEnd");
ci(Od, "onAnimationIteration");
ci(Pd, "onAnimationStart");
ci("dblclick", "onDoubleClick");
ci("focusin", "onFocus");
ci("focusout", "onBlur");
ci(Md, "onTransitionEnd");
rr("onMouseEnter", ["mouseout", "mouseover"]);
rr("onMouseLeave", ["mouseout", "mouseover"]);
rr("onPointerEnter", ["pointerout", "pointerover"]);
rr("onPointerLeave", ["pointerout", "pointerover"]);
Ai("onChange", "change click focusin focusout input keydown keyup selectionchange".split(" "));
Ai("onSelect", "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));
Ai("onBeforeInput", ["compositionend", "keypress", "textInput", "paste"]);
Ai("onCompositionEnd", "compositionend focusout keydown keypress keyup mousedown".split(" "));
Ai("onCompositionStart", "compositionstart focusout keydown keypress keyup mousedown".split(" "));
Ai("onCompositionUpdate", "compositionupdate focusout keydown keypress keyup mousedown".split(" "));
var Ir = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "), R1 = new Set("cancel close invalid load scroll toggle".split(" ").concat(Ir));
function $u(t, e, n) {
  var o = t.type || "unknown-event";
  t.currentTarget = n, Fm(o, e, void 0, t), t.currentTarget = null;
}
function Dd(t, e) {
  e = (e & 4) !== 0;
  for (var n = 0; n < t.length; n++) {
    var o = t[n], d = o.event;
    o = o.listeners;
    t: {
      var g = void 0;
      if (e) for (var b = o.length - 1; 0 <= b; b--) {
        var _ = o[b], w = _.instance, E = _.currentTarget;
        if (_ = _.listener, w !== g && d.isPropagationStopped()) break t;
        $u(d, _, E), g = w;
      }
      else for (b = 0; b < o.length; b++) {
        if (_ = o[b], w = _.instance, E = _.currentTarget, _ = _.listener, w !== g && d.isPropagationStopped()) break t;
        $u(d, _, E), g = w;
      }
    }
  }
  if (uo) throw t = ul, uo = !1, ul = null, t;
}
function Gt(t, e) {
  var n = e[xl];
  n === void 0 && (n = e[xl] = /* @__PURE__ */ new Set());
  var o = t + "__bubble";
  n.has(o) || (jd(e, t, 2, !1), n.add(o));
}
function Es(t, e, n) {
  var o = 0;
  e && (o |= 4), jd(n, t, o, e);
}
var La = "_reactListening" + Math.random().toString(36).slice(2);
function ea(t) {
  if (!t[La]) {
    t[La] = !0, Nf.forEach(function (n) {
      n !== "selectionchange" && (R1.has(n) || Es(n, !1, t), Es(n, !0, t));
    });
    var e = t.nodeType === 9 ? t : t.ownerDocument;
    e === null || e[La] || (e[La] = !0, Es("selectionchange", !1, e));
  }
}
function jd(t, e, n, o) {
  switch (md(e)) {
    case 1:
      var d = Km;
      break;
    case 4:
      d = Jm;
      break;
    default:
      d = sc;
  }
  n = d.bind(null, e, n, t), d = void 0, !cl || e !== "touchstart" && e !== "touchmove" && e !== "wheel" || (d = !0), o ? d !== void 0 ? t.addEventListener(e, n, { capture: !0, passive: d }) : t.addEventListener(e, n, !0) : d !== void 0 ? t.addEventListener(e, n, { passive: d }) : t.addEventListener(e, n, !1);
}
function ks(t, e, n, o, d) {
  var g = o;
  if (!(e & 1) && !(e & 2) && o !== null) t: for (; ;) {
    if (o === null) return;
    var b = o.tag;
    if (b === 3 || b === 4) {
      var _ = o.stateNode.containerInfo;
      if (_ === d || _.nodeType === 8 && _.parentNode === d) break;
      if (b === 4) for (b = o.return; b !== null;) {
        var w = b.tag;
        if ((w === 3 || w === 4) && (w = b.stateNode.containerInfo, w === d || w.nodeType === 8 && w.parentNode === d)) return;
        b = b.return;
      }
      for (; _ !== null;) {
        if (b = _i(_), b === null) return;
        if (w = b.tag, w === 5 || w === 6) {
          o = g = b;
          continue t;
        }
        _ = _.parentNode;
      }
    }
    o = o.return;
  }
  nd(function () {
    var E = g, j = ic(n), H = [];
    t: {
      var G = Ad.get(t);
      if (G !== void 0) {
        var J = cc, nt = t;
        switch (t) {
          case "keypress":
            if ($a(n) === 0) break t;
          case "keydown":
          case "keyup":
            J = d1;
            break;
          case "focusin":
            nt = "focus", J = bs;
            break;
          case "focusout":
            nt = "blur", J = bs;
            break;
          case "beforeblur":
          case "afterblur":
            J = bs;
            break;
          case "click":
            if (n.button === 2) break t;
          case "auxclick":
          case "dblclick":
          case "mousedown":
          case "mousemove":
          case "mouseup":
          case "mouseout":
          case "mouseover":
          case "contextmenu":
            J = Uu;
            break;
          case "drag":
          case "dragend":
          case "dragenter":
          case "dragexit":
          case "dragleave":
          case "dragover":
          case "dragstart":
          case "drop":
            J = e1;
            break;
          case "touchcancel":
          case "touchend":
          case "touchmove":
          case "touchstart":
            J = m1;
            break;
          case Td:
          case Od:
          case Pd:
            J = r1;
            break;
          case Md:
            J = y1;
            break;
          case "scroll":
            J = $m;
            break;
          case "wheel":
            J = x1;
            break;
          case "copy":
          case "cut":
          case "paste":
            J = o1;
            break;
          case "gotpointercapture":
          case "lostpointercapture":
          case "pointercancel":
          case "pointerdown":
          case "pointermove":
          case "pointerout":
          case "pointerover":
          case "pointerup":
            J = Wu;
        }
        var rt = (e & 4) !== 0, vt = !rt && t === "scroll", Y = rt ? G !== null ? G + "Capture" : null : G;
        rt = [];
        for (var L = E, A; L !== null;) {
          A = L;
          var tt = A.stateNode;
          if (A.tag === 5 && tt !== null && (A = tt, Y !== null && (tt = Zr(L, Y), tt != null && rt.push(na(L, tt, A)))), vt) break;
          L = L.return;
        }
        0 < rt.length && (G = new J(G, nt, null, n, j), H.push({ event: G, listeners: rt }));
      }
    }
    if (!(e & 7)) {
      t: {
        if (G = t === "mouseover" || t === "pointerover", J = t === "mouseout" || t === "pointerout", G && n !== sl && (nt = n.relatedTarget || n.fromElement) && (_i(nt) || nt[Un])) break t;
        if ((J || G) && (G = j.window === j ? j : (G = j.ownerDocument) ? G.defaultView || G.parentWindow : window, J ? (nt = n.relatedTarget || n.toElement, J = E, nt = nt ? _i(nt) : null, nt !== null && (vt = Di(nt), nt !== vt || nt.tag !== 5 && nt.tag !== 6) && (nt = null)) : (J = null, nt = E), J !== nt)) {
          if (rt = Uu, tt = "onMouseLeave", Y = "onMouseEnter", L = "mouse", (t === "pointerout" || t === "pointerover") && (rt = Wu, tt = "onPointerLeave", Y = "onPointerEnter", L = "pointer"), vt = J == null ? G : Xi(J), A = nt == null ? G : Xi(nt), G = new rt(tt, L + "leave", J, n, j), G.target = vt, G.relatedTarget = A, tt = null, _i(j) === E && (rt = new rt(Y, L + "enter", nt, n, j), rt.target = A, rt.relatedTarget = vt, tt = rt), vt = tt, J && nt) e: {
            for (rt = J, Y = nt, L = 0, A = rt; A; A = Ri(A)) L++;
            for (A = 0, tt = Y; tt; tt = Ri(tt)) A++;
            for (; 0 < L - A;) rt = Ri(rt), L--;
            for (; 0 < A - L;) Y = Ri(Y), A--;
            for (; L--;) {
              if (rt === Y || Y !== null && rt === Y.alternate) break e;
              rt = Ri(rt), Y = Ri(Y);
            }
            rt = null;
          }
          else rt = null;
          J !== null && th(H, G, J, rt, !1), nt !== null && vt !== null && th(H, vt, nt, rt, !0);
        }
      }
      t: {
        if (G = E ? Xi(E) : window, J = G.nodeName && G.nodeName.toLowerCase(), J === "select" || J === "input" && G.type === "file") var dt = T1;
        else if (Vu(G)) if (wd) dt = A1;
        else {
          dt = P1;
          var it = O1;
        }
        else (J = G.nodeName) && J.toLowerCase() === "input" && (G.type === "checkbox" || G.type === "radio") && (dt = M1);
        if (dt && (dt = dt(t, E))) {
          _d(H, dt, n, j);
          break t;
        }
        it && it(t, G, E), t === "focusout" && (it = G._wrapperState) && it.controlled && G.type === "number" && nl(G, "number", G.value);
      }
      switch (it = E ? Xi(E) : window, t) {
        case "focusin":
          (Vu(it) || it.contentEditable === "true") && (Hi = it, pl = E, Ur = null);
          break;
        case "focusout":
          Ur = pl = Hi = null;
          break;
        case "mousedown":
          gl = !0;
          break;
        case "contextmenu":
        case "mouseup":
        case "dragend":
          gl = !1, Ku(H, n, j);
          break;
        case "selectionchange":
          if (L1) break;
        case "keydown":
        case "keyup":
          Ku(H, n, j);
      }
      var ht;
      if (hc) t: {
        switch (t) {
          case "compositionstart":
            var v = "onCompositionStart";
            break t;
          case "compositionend":
            v = "onCompositionEnd";
            break t;
          case "compositionupdate":
            v = "onCompositionUpdate";
            break t;
        }
        v = void 0;
      }
      else Ui ? bd(t, n) && (v = "onCompositionEnd") : t === "keydown" && n.keyCode === 229 && (v = "onCompositionStart");
      v && (yd && n.locale !== "ko" && (Ui || v !== "onCompositionStart" ? v === "onCompositionEnd" && Ui && (ht = vd()) : (Kn = j, lc = "value" in Kn ? Kn.value : Kn.textContent, Ui = !0)), it = mo(E, v), 0 < it.length && (v = new Hu(v, t, null, n, j), H.push({ event: v, listeners: it }), ht ? v.data = ht : (ht = xd(n), ht !== null && (v.data = ht)))), (ht = w1 ? C1(t, n) : S1(t, n)) && (E = mo(E, "onBeforeInput"), 0 < E.length && (j = new Hu("onBeforeInput", "beforeinput", null, n, j), H.push({ event: j, listeners: E }), j.data = ht));
    }
    Dd(H, e);
  });
}
function na(t, e, n) {
  return { instance: t, listener: e, currentTarget: n };
}
function mo(t, e) {
  for (var n = e + "Capture", o = []; t !== null;) {
    var d = t, g = d.stateNode;
    d.tag === 5 && g !== null && (d = g, g = Zr(t, n), g != null && o.unshift(na(t, g, d)), g = Zr(t, e), g != null && o.push(na(t, g, d))), t = t.return;
  }
  return o;
}
function Ri(t) {
  if (t === null) return null;
  do
    t = t.return;
  while (t && t.tag !== 5);
  return t || null;
}
function th(t, e, n, o, d) {
  for (var g = e._reactName, b = []; n !== null && n !== o;) {
    var _ = n, w = _.alternate, E = _.stateNode;
    if (w !== null && w === o) break;
    _.tag === 5 && E !== null && (_ = E, d ? (w = Zr(n, g), w != null && b.unshift(na(n, w, _))) : d || (w = Zr(n, g), w != null && b.push(na(n, w, _)))), n = n.return;
  }
  b.length !== 0 && t.push({ event: e, listeners: b });
}
var B1 = /\r\n?/g, z1 = /\u0000|\uFFFD/g;
function eh(t) {
  return (typeof t == "string" ? t : "" + t).replace(B1, `
`).replace(z1, "");
}
function Fa(t, e, n) {
  if (e = eh(e), eh(t) !== e && n) throw Error(pt(425));
}
function vo() {
}
var ml = null, vl = null;
function yl(t, e) {
  return t === "textarea" || t === "noscript" || typeof e.children == "string" || typeof e.children == "number" || typeof e.dangerouslySetInnerHTML == "object" && e.dangerouslySetInnerHTML !== null && e.dangerouslySetInnerHTML.__html != null;
}
var bl = typeof setTimeout == "function" ? setTimeout : void 0, N1 = typeof clearTimeout == "function" ? clearTimeout : void 0, nh = typeof Promise == "function" ? Promise : void 0, U1 = typeof queueMicrotask == "function" ? queueMicrotask : typeof nh < "u" ? function (t) {
  return nh.resolve(null).then(t).catch(H1);
} : bl;
function H1(t) {
  setTimeout(function () {
    throw t;
  });
}
function Ts(t, e) {
  var n = e, o = 0;
  do {
    var d = n.nextSibling;
    if (t.removeChild(n), d && d.nodeType === 8) if (n = d.data, n === "/$") {
      if (o === 0) {
        t.removeChild(d), Jr(e);
        return;
      }
      o--;
    } else n !== "$" && n !== "$?" && n !== "$!" || o++;
    n = d;
  } while (n);
  Jr(e);
}
function ni(t) {
  for (; t != null; t = t.nextSibling) {
    var e = t.nodeType;
    if (e === 1 || e === 3) break;
    if (e === 8) {
      if (e = t.data, e === "$" || e === "$!" || e === "$?") break;
      if (e === "/$") return null;
    }
  }
  return t;
}
function ih(t) {
  t = t.previousSibling;
  for (var e = 0; t;) {
    if (t.nodeType === 8) {
      var n = t.data;
      if (n === "$" || n === "$!" || n === "$?") {
        if (e === 0) return t;
        e--;
      } else n === "/$" && e++;
    }
    t = t.previousSibling;
  }
  return null;
}
var gr = Math.random().toString(36).slice(2), En = "__reactFiber$" + gr, ia = "__reactProps$" + gr, Un = "__reactContainer$" + gr, xl = "__reactEvents$" + gr, W1 = "__reactListeners$" + gr, X1 = "__reactHandles$" + gr;
function _i(t) {
  var e = t[En];
  if (e) return e;
  for (var n = t.parentNode; n;) {
    if (e = n[Un] || n[En]) {
      if (n = e.alternate, e.child !== null || n !== null && n.child !== null) for (t = ih(t); t !== null;) {
        if (n = t[En]) return n;
        t = ih(t);
      }
      return e;
    }
    t = n, n = t.parentNode;
  }
  return null;
}
function pa(t) {
  return t = t[En] || t[Un], !t || t.tag !== 5 && t.tag !== 6 && t.tag !== 13 && t.tag !== 3 ? null : t;
}
function Xi(t) {
  if (t.tag === 5 || t.tag === 6) return t.stateNode;
  throw Error(pt(33));
}
function Xo(t) {
  return t[ia] || null;
}
var _l = [], Yi = -1;
function ui(t) {
  return { current: t };
}
function qt(t) {
  0 > Yi || (t.current = _l[Yi], _l[Yi] = null, Yi--);
}
function Vt(t, e) {
  Yi++, _l[Yi] = t.current, t.current = e;
}
var li = {}, _e = ui(li), Le = ui(!1), ki = li;
function ar(t, e) {
  var n = t.type.contextTypes;
  if (!n) return li;
  var o = t.stateNode;
  if (o && o.__reactInternalMemoizedUnmaskedChildContext === e) return o.__reactInternalMemoizedMaskedChildContext;
  var d = {}, g;
  for (g in n) d[g] = e[g];
  return o && (t = t.stateNode, t.__reactInternalMemoizedUnmaskedChildContext = e, t.__reactInternalMemoizedMaskedChildContext = d), d;
}
function Fe(t) {
  return t = t.childContextTypes, t != null;
}
function yo() {
  qt(Le), qt(_e);
}
function rh(t, e, n) {
  if (_e.current !== li) throw Error(pt(168));
  Vt(_e, e), Vt(Le, n);
}
function Ld(t, e, n) {
  var o = t.stateNode;
  if (e = e.childContextTypes, typeof o.getChildContext != "function") return n;
  o = o.getChildContext();
  for (var d in o) if (!(d in e)) throw Error(pt(108, Om(t) || "Unknown", d));
  return ne({}, n, o);
}
function bo(t) {
  return t = (t = t.stateNode) && t.__reactInternalMemoizedMergedChildContext || li, ki = _e.current, Vt(_e, t), Vt(Le, Le.current), !0;
}
function ah(t, e, n) {
  var o = t.stateNode;
  if (!o) throw Error(pt(169));
  n ? (t = Ld(t, e, ki), o.__reactInternalMemoizedMergedChildContext = t, qt(Le), qt(_e), Vt(_e, t)) : qt(Le), Vt(Le, n);
}
var Fn = null, Yo = !1, Os = !1;
function Fd(t) {
  Fn === null ? Fn = [t] : Fn.push(t);
}
function Y1(t) {
  Yo = !0, Fd(t);
}
function hi() {
  if (!Os && Fn !== null) {
    Os = !0;
    var t = 0, e = Xt;
    try {
      var n = Fn;
      for (Xt = 1; t < n.length; t++) {
        var o = n[t];
        do
          o = o(!0);
        while (o !== null);
      }
      Fn = null, Yo = !1;
    } catch (d) {
      throw Fn !== null && (Fn = Fn.slice(t + 1)), od(rc, hi), d;
    } finally {
      Xt = e, Os = !1;
    }
  }
  return null;
}
var Vi = [], Gi = 0, xo = null, _o = 0, Qe = [], Ke = 0, Ti = null, In = 1, Rn = "";
function vi(t, e) {
  Vi[Gi++] = _o, Vi[Gi++] = xo, xo = t, _o = e;
}
function Id(t, e, n) {
  Qe[Ke++] = In, Qe[Ke++] = Rn, Qe[Ke++] = Ti, Ti = t;
  var o = In;
  t = Rn;
  var d = 32 - mn(o) - 1;
  o &= ~(1 << d), n += 1;
  var g = 32 - mn(e) + d;
  if (30 < g) {
    var b = d - d % 5;
    g = (o & (1 << b) - 1).toString(32), o >>= b, d -= b, In = 1 << 32 - mn(e) + d | n << d | o, Rn = g + t;
  } else In = 1 << g | n << d | o, Rn = t;
}
function dc(t) {
  t.return !== null && (vi(t, 1), Id(t, 1, 0));
}
function pc(t) {
  for (; t === xo;) xo = Vi[--Gi], Vi[Gi] = null, _o = Vi[--Gi], Vi[Gi] = null;
  for (; t === Ti;) Ti = Qe[--Ke], Qe[Ke] = null, Rn = Qe[--Ke], Qe[Ke] = null, In = Qe[--Ke], Qe[Ke] = null;
}
var Ue = null, Ne = null, Qt = !1, dn = null;
function Rd(t, e) {
  var n = $e(5, null, null, 0);
  n.elementType = "DELETED", n.stateNode = e, n.return = t, e = t.deletions, e === null ? (t.deletions = [n], t.flags |= 16) : e.push(n);
}
function oh(t, e) {
  switch (t.tag) {
    case 5:
      var n = t.type;
      return e = e.nodeType !== 1 || n.toLowerCase() !== e.nodeName.toLowerCase() ? null : e, e !== null ? (t.stateNode = e, Ue = t, Ne = ni(e.firstChild), !0) : !1;
    case 6:
      return e = t.pendingProps === "" || e.nodeType !== 3 ? null : e, e !== null ? (t.stateNode = e, Ue = t, Ne = null, !0) : !1;
    case 13:
      return e = e.nodeType !== 8 ? null : e, e !== null ? (n = Ti !== null ? { id: In, overflow: Rn } : null, t.memoizedState = { dehydrated: e, treeContext: n, retryLane: 1073741824 }, n = $e(18, null, null, 0), n.stateNode = e, n.return = t, t.child = n, Ue = t, Ne = null, !0) : !1;
    default:
      return !1;
  }
}
function wl(t) {
  return (t.mode & 1) !== 0 && (t.flags & 128) === 0;
}
function Cl(t) {
  if (Qt) {
    var e = Ne;
    if (e) {
      var n = e;
      if (!oh(t, e)) {
        if (wl(t)) throw Error(pt(418));
        e = ni(n.nextSibling);
        var o = Ue;
        e && oh(t, e) ? Rd(o, n) : (t.flags = t.flags & -4097 | 2, Qt = !1, Ue = t);
      }
    } else {
      if (wl(t)) throw Error(pt(418));
      t.flags = t.flags & -4097 | 2, Qt = !1, Ue = t;
    }
  }
}
function sh(t) {
  for (t = t.return; t !== null && t.tag !== 5 && t.tag !== 3 && t.tag !== 13;) t = t.return;
  Ue = t;
}
function Ia(t) {
  if (t !== Ue) return !1;
  if (!Qt) return sh(t), Qt = !0, !1;
  var e;
  if ((e = t.tag !== 3) && !(e = t.tag !== 5) && (e = t.type, e = e !== "head" && e !== "body" && !yl(t.type, t.memoizedProps)), e && (e = Ne)) {
    if (wl(t)) throw Bd(), Error(pt(418));
    for (; e;) Rd(t, e), e = ni(e.nextSibling);
  }
  if (sh(t), t.tag === 13) {
    if (t = t.memoizedState, t = t !== null ? t.dehydrated : null, !t) throw Error(pt(317));
    t: {
      for (t = t.nextSibling, e = 0; t;) {
        if (t.nodeType === 8) {
          var n = t.data;
          if (n === "/$") {
            if (e === 0) {
              Ne = ni(t.nextSibling);
              break t;
            }
            e--;
          } else n !== "$" && n !== "$!" && n !== "$?" || e++;
        }
        t = t.nextSibling;
      }
      Ne = null;
    }
  } else Ne = Ue ? ni(t.stateNode.nextSibling) : null;
  return !0;
}
function Bd() {
  for (var t = Ne; t;) t = ni(t.nextSibling);
}
function or() {
  Ne = Ue = null, Qt = !1;
}
function gc(t) {
  dn === null ? dn = [t] : dn.push(t);
}
var V1 = Xn.ReactCurrentBatchConfig;
function Mr(t, e, n) {
  if (t = n.ref, t !== null && typeof t != "function" && typeof t != "object") {
    if (n._owner) {
      if (n = n._owner, n) {
        if (n.tag !== 1) throw Error(pt(309));
        var o = n.stateNode;
      }
      if (!o) throw Error(pt(147, t));
      var d = o, g = "" + t;
      return e !== null && e.ref !== null && typeof e.ref == "function" && e.ref._stringRef === g ? e.ref : (e = function (b) {
        var _ = d.refs;
        b === null ? delete _[g] : _[g] = b;
      }, e._stringRef = g, e);
    }
    if (typeof t != "string") throw Error(pt(284));
    if (!n._owner) throw Error(pt(290, t));
  }
  return t;
}
function Ra(t, e) {
  throw t = Object.prototype.toString.call(e), Error(pt(31, t === "[object Object]" ? "object with keys {" + Object.keys(e).join(", ") + "}" : t));
}
function lh(t) {
  var e = t._init;
  return e(t._payload);
}
function zd(t) {
  function e(Y, L) {
    if (t) {
      var A = Y.deletions;
      A === null ? (Y.deletions = [L], Y.flags |= 16) : A.push(L);
    }
  }
  function n(Y, L) {
    if (!t) return null;
    for (; L !== null;) e(Y, L), L = L.sibling;
    return null;
  }
  function o(Y, L) {
    for (Y = /* @__PURE__ */ new Map(); L !== null;) L.key !== null ? Y.set(L.key, L) : Y.set(L.index, L), L = L.sibling;
    return Y;
  }
  function d(Y, L) {
    return Y = oi(Y, L), Y.index = 0, Y.sibling = null, Y;
  }
  function g(Y, L, A) {
    return Y.index = A, t ? (A = Y.alternate, A !== null ? (A = A.index, A < L ? (Y.flags |= 2, L) : A) : (Y.flags |= 2, L)) : (Y.flags |= 1048576, L);
  }
  function b(Y) {
    return t && Y.alternate === null && (Y.flags |= 2), Y;
  }
  function _(Y, L, A, tt) {
    return L === null || L.tag !== 6 ? (L = Fs(A, Y.mode, tt), L.return = Y, L) : (L = d(L, A), L.return = Y, L);
  }
  function w(Y, L, A, tt) {
    var dt = A.type;
    return dt === Ni ? j(Y, L, A.props.children, tt, A.key) : L !== null && (L.elementType === dt || typeof dt == "object" && dt !== null && dt.$$typeof === Vn && lh(dt) === L.type) ? (tt = d(L, A.props), tt.ref = Mr(Y, L, A), tt.return = Y, tt) : (tt = oo(A.type, A.key, A.props, null, Y.mode, tt), tt.ref = Mr(Y, L, A), tt.return = Y, tt);
  }
  function E(Y, L, A, tt) {
    return L === null || L.tag !== 4 || L.stateNode.containerInfo !== A.containerInfo || L.stateNode.implementation !== A.implementation ? (L = Is(A, Y.mode, tt), L.return = Y, L) : (L = d(L, A.children || []), L.return = Y, L);
  }
  function j(Y, L, A, tt, dt) {
    return L === null || L.tag !== 7 ? (L = Ei(A, Y.mode, tt, dt), L.return = Y, L) : (L = d(L, A), L.return = Y, L);
  }
  function H(Y, L, A) {
    if (typeof L == "string" && L !== "" || typeof L == "number") return L = Fs("" + L, Y.mode, A), L.return = Y, L;
    if (typeof L == "object" && L !== null) {
      switch (L.$$typeof) {
        case ka:
          return A = oo(L.type, L.key, L.props, null, Y.mode, A), A.ref = Mr(Y, null, L), A.return = Y, A;
        case zi:
          return L = Is(L, Y.mode, A), L.return = Y, L;
        case Vn:
          var tt = L._init;
          return H(Y, tt(L._payload), A);
      }
      if (Lr(L) || Er(L)) return L = Ei(L, Y.mode, A, null), L.return = Y, L;
      Ra(Y, L);
    }
    return null;
  }
  function G(Y, L, A, tt) {
    var dt = L !== null ? L.key : null;
    if (typeof A == "string" && A !== "" || typeof A == "number") return dt !== null ? null : _(Y, L, "" + A, tt);
    if (typeof A == "object" && A !== null) {
      switch (A.$$typeof) {
        case ka:
          return A.key === dt ? w(Y, L, A, tt) : null;
        case zi:
          return A.key === dt ? E(Y, L, A, tt) : null;
        case Vn:
          return dt = A._init, G(
            Y,
            L,
            dt(A._payload),
            tt
          );
      }
      if (Lr(A) || Er(A)) return dt !== null ? null : j(Y, L, A, tt, null);
      Ra(Y, A);
    }
    return null;
  }
  function J(Y, L, A, tt, dt) {
    if (typeof tt == "string" && tt !== "" || typeof tt == "number") return Y = Y.get(A) || null, _(L, Y, "" + tt, dt);
    if (typeof tt == "object" && tt !== null) {
      switch (tt.$$typeof) {
        case ka:
          return Y = Y.get(tt.key === null ? A : tt.key) || null, w(L, Y, tt, dt);
        case zi:
          return Y = Y.get(tt.key === null ? A : tt.key) || null, E(L, Y, tt, dt);
        case Vn:
          var it = tt._init;
          return J(Y, L, A, it(tt._payload), dt);
      }
      if (Lr(tt) || Er(tt)) return Y = Y.get(A) || null, j(L, Y, tt, dt, null);
      Ra(L, tt);
    }
    return null;
  }
  function nt(Y, L, A, tt) {
    for (var dt = null, it = null, ht = L, v = L = 0, Ct = null; ht !== null && v < A.length; v++) {
      ht.index > v ? (Ct = ht, ht = null) : Ct = ht.sibling;
      var F = G(Y, ht, A[v], tt);
      if (F === null) {
        ht === null && (ht = Ct);
        break;
      }
      t && ht && F.alternate === null && e(Y, ht), L = g(F, L, v), it === null ? dt = F : it.sibling = F, it = F, ht = Ct;
    }
    if (v === A.length) return n(Y, ht), Qt && vi(Y, v), dt;
    if (ht === null) {
      for (; v < A.length; v++) ht = H(Y, A[v], tt), ht !== null && (L = g(ht, L, v), it === null ? dt = ht : it.sibling = ht, it = ht);
      return Qt && vi(Y, v), dt;
    }
    for (ht = o(Y, ht); v < A.length; v++) Ct = J(ht, Y, v, A[v], tt), Ct !== null && (t && Ct.alternate !== null && ht.delete(Ct.key === null ? v : Ct.key), L = g(Ct, L, v), it === null ? dt = Ct : it.sibling = Ct, it = Ct);
    return t && ht.forEach(function (Et) {
      return e(Y, Et);
    }), Qt && vi(Y, v), dt;
  }
  function rt(Y, L, A, tt) {
    var dt = Er(A);
    if (typeof dt != "function") throw Error(pt(150));
    if (A = dt.call(A), A == null) throw Error(pt(151));
    for (var it = dt = null, ht = L, v = L = 0, Ct = null, F = A.next(); ht !== null && !F.done; v++, F = A.next()) {
      ht.index > v ? (Ct = ht, ht = null) : Ct = ht.sibling;
      var Et = G(Y, ht, F.value, tt);
      if (Et === null) {
        ht === null && (ht = Ct);
        break;
      }
      t && ht && Et.alternate === null && e(Y, ht), L = g(Et, L, v), it === null ? dt = Et : it.sibling = Et, it = Et, ht = Ct;
    }
    if (F.done) return n(
      Y,
      ht
    ), Qt && vi(Y, v), dt;
    if (ht === null) {
      for (; !F.done; v++, F = A.next()) F = H(Y, F.value, tt), F !== null && (L = g(F, L, v), it === null ? dt = F : it.sibling = F, it = F);
      return Qt && vi(Y, v), dt;
    }
    for (ht = o(Y, ht); !F.done; v++, F = A.next()) F = J(ht, Y, v, F.value, tt), F !== null && (t && F.alternate !== null && ht.delete(F.key === null ? v : F.key), L = g(F, L, v), it === null ? dt = F : it.sibling = F, it = F);
    return t && ht.forEach(function (Ht) {
      return e(Y, Ht);
    }), Qt && vi(Y, v), dt;
  }
  function vt(Y, L, A, tt) {
    if (typeof A == "object" && A !== null && A.type === Ni && A.key === null && (A = A.props.children), typeof A == "object" && A !== null) {
      switch (A.$$typeof) {
        case ka:
          t: {
            for (var dt = A.key, it = L; it !== null;) {
              if (it.key === dt) {
                if (dt = A.type, dt === Ni) {
                  if (it.tag === 7) {
                    n(Y, it.sibling), L = d(it, A.props.children), L.return = Y, Y = L;
                    break t;
                  }
                } else if (it.elementType === dt || typeof dt == "object" && dt !== null && dt.$$typeof === Vn && lh(dt) === it.type) {
                  n(Y, it.sibling), L = d(it, A.props), L.ref = Mr(Y, it, A), L.return = Y, Y = L;
                  break t;
                }
                n(Y, it);
                break;
              } else e(Y, it);
              it = it.sibling;
            }
            A.type === Ni ? (L = Ei(A.props.children, Y.mode, tt, A.key), L.return = Y, Y = L) : (tt = oo(A.type, A.key, A.props, null, Y.mode, tt), tt.ref = Mr(Y, L, A), tt.return = Y, Y = tt);
          }
          return b(Y);
        case zi:
          t: {
            for (it = A.key; L !== null;) {
              if (L.key === it) if (L.tag === 4 && L.stateNode.containerInfo === A.containerInfo && L.stateNode.implementation === A.implementation) {
                n(Y, L.sibling), L = d(L, A.children || []), L.return = Y, Y = L;
                break t;
              } else {
                n(Y, L);
                break;
              }
              else e(Y, L);
              L = L.sibling;
            }
            L = Is(A, Y.mode, tt), L.return = Y, Y = L;
          }
          return b(Y);
        case Vn:
          return it = A._init, vt(Y, L, it(A._payload), tt);
      }
      if (Lr(A)) return nt(Y, L, A, tt);
      if (Er(A)) return rt(Y, L, A, tt);
      Ra(Y, A);
    }
    return typeof A == "string" && A !== "" || typeof A == "number" ? (A = "" + A, L !== null && L.tag === 6 ? (n(Y, L.sibling), L = d(L, A), L.return = Y, Y = L) : (n(Y, L), L = Fs(A, Y.mode, tt), L.return = Y, Y = L), b(Y)) : n(Y, L);
  }
  return vt;
}
var sr = zd(!0), Nd = zd(!1), wo = ui(null), Co = null, qi = null, mc = null;
function vc() {
  mc = qi = Co = null;
}
function yc(t) {
  var e = wo.current;
  qt(wo), t._currentValue = e;
}
function Sl(t, e, n) {
  for (; t !== null;) {
    var o = t.alternate;
    if ((t.childLanes & e) !== e ? (t.childLanes |= e, o !== null && (o.childLanes |= e)) : o !== null && (o.childLanes & e) !== e && (o.childLanes |= e), t === n) break;
    t = t.return;
  }
}
function nr(t, e) {
  Co = t, mc = qi = null, t = t.dependencies, t !== null && t.firstContext !== null && (t.lanes & e && (je = !0), t.firstContext = null);
}
function rn(t) {
  var e = t._currentValue;
  if (mc !== t) if (t = { context: t, memoizedValue: e, next: null }, qi === null) {
    if (Co === null) throw Error(pt(308));
    qi = t, Co.dependencies = { lanes: 0, firstContext: t };
  } else qi = qi.next = t;
  return e;
}
var wi = null;
function bc(t) {
  wi === null ? wi = [t] : wi.push(t);
}
function Ud(t, e, n, o) {
  var d = e.interleaved;
  return d === null ? (n.next = n, bc(e)) : (n.next = d.next, d.next = n), e.interleaved = n, Hn(t, o);
}
function Hn(t, e) {
  t.lanes |= e;
  var n = t.alternate;
  for (n !== null && (n.lanes |= e), n = t, t = t.return; t !== null;) t.childLanes |= e, n = t.alternate, n !== null && (n.childLanes |= e), n = t, t = t.return;
  return n.tag === 3 ? n.stateNode : null;
}
var Gn = !1;
function xc(t) {
  t.updateQueue = { baseState: t.memoizedState, firstBaseUpdate: null, lastBaseUpdate: null, shared: { pending: null, interleaved: null, lanes: 0 }, effects: null };
}
function Hd(t, e) {
  t = t.updateQueue, e.updateQueue === t && (e.updateQueue = { baseState: t.baseState, firstBaseUpdate: t.firstBaseUpdate, lastBaseUpdate: t.lastBaseUpdate, shared: t.shared, effects: t.effects });
}
function zn(t, e) {
  return { eventTime: t, lane: e, tag: 0, payload: null, callback: null, next: null };
}
function ii(t, e, n) {
  var o = t.updateQueue;
  if (o === null) return null;
  if (o = o.shared, Ut & 2) {
    var d = o.pending;
    return d === null ? e.next = e : (e.next = d.next, d.next = e), o.pending = e, Hn(t, n);
  }
  return d = o.interleaved, d === null ? (e.next = e, bc(o)) : (e.next = d.next, d.next = e), o.interleaved = e, Hn(t, n);
}
function to(t, e, n) {
  if (e = e.updateQueue, e !== null && (e = e.shared, (n & 4194240) !== 0)) {
    var o = e.lanes;
    o &= t.pendingLanes, n |= o, e.lanes = n, ac(t, n);
  }
}
function ch(t, e) {
  var n = t.updateQueue, o = t.alternate;
  if (o !== null && (o = o.updateQueue, n === o)) {
    var d = null, g = null;
    if (n = n.firstBaseUpdate, n !== null) {
      do {
        var b = { eventTime: n.eventTime, lane: n.lane, tag: n.tag, payload: n.payload, callback: n.callback, next: null };
        g === null ? d = g = b : g = g.next = b, n = n.next;
      } while (n !== null);
      g === null ? d = g = e : g = g.next = e;
    } else d = g = e;
    n = { baseState: o.baseState, firstBaseUpdate: d, lastBaseUpdate: g, shared: o.shared, effects: o.effects }, t.updateQueue = n;
    return;
  }
  t = n.lastBaseUpdate, t === null ? n.firstBaseUpdate = e : t.next = e, n.lastBaseUpdate = e;
}
function So(t, e, n, o) {
  var d = t.updateQueue;
  Gn = !1;
  var g = d.firstBaseUpdate, b = d.lastBaseUpdate, _ = d.shared.pending;
  if (_ !== null) {
    d.shared.pending = null;
    var w = _, E = w.next;
    w.next = null, b === null ? g = E : b.next = E, b = w;
    var j = t.alternate;
    j !== null && (j = j.updateQueue, _ = j.lastBaseUpdate, _ !== b && (_ === null ? j.firstBaseUpdate = E : _.next = E, j.lastBaseUpdate = w));
  }
  if (g !== null) {
    var H = d.baseState;
    b = 0, j = E = w = null, _ = g;
    do {
      var G = _.lane, J = _.eventTime;
      if ((o & G) === G) {
        j !== null && (j = j.next = {
          eventTime: J,
          lane: 0,
          tag: _.tag,
          payload: _.payload,
          callback: _.callback,
          next: null
        });
        t: {
          var nt = t, rt = _;
          switch (G = e, J = n, rt.tag) {
            case 1:
              if (nt = rt.payload, typeof nt == "function") {
                H = nt.call(J, H, G);
                break t;
              }
              H = nt;
              break t;
            case 3:
              nt.flags = nt.flags & -65537 | 128;
            case 0:
              if (nt = rt.payload, G = typeof nt == "function" ? nt.call(J, H, G) : nt, G == null) break t;
              H = ne({}, H, G);
              break t;
            case 2:
              Gn = !0;
          }
        }
        _.callback !== null && _.lane !== 0 && (t.flags |= 64, G = d.effects, G === null ? d.effects = [_] : G.push(_));
      } else J = { eventTime: J, lane: G, tag: _.tag, payload: _.payload, callback: _.callback, next: null }, j === null ? (E = j = J, w = H) : j = j.next = J, b |= G;
      if (_ = _.next, _ === null) {
        if (_ = d.shared.pending, _ === null) break;
        G = _, _ = G.next, G.next = null, d.lastBaseUpdate = G, d.shared.pending = null;
      }
    } while (!0);
    if (j === null && (w = H), d.baseState = w, d.firstBaseUpdate = E, d.lastBaseUpdate = j, e = d.shared.interleaved, e !== null) {
      d = e;
      do
        b |= d.lane, d = d.next;
      while (d !== e);
    } else g === null && (d.shared.lanes = 0);
    Pi |= b, t.lanes = b, t.memoizedState = H;
  }
}
function uh(t, e, n) {
  if (t = e.effects, e.effects = null, t !== null) for (e = 0; e < t.length; e++) {
    var o = t[e], d = o.callback;
    if (d !== null) {
      if (o.callback = null, o = n, typeof d != "function") throw Error(pt(191, d));
      d.call(o);
    }
  }
}
var ga = {}, An = ui(ga), ra = ui(ga), aa = ui(ga);
function Ci(t) {
  if (t === ga) throw Error(pt(174));
  return t;
}
function _c(t, e) {
  switch (Vt(aa, e), Vt(ra, t), Vt(An, ga), t = e.nodeType, t) {
    case 9:
    case 11:
      e = (e = e.documentElement) ? e.namespaceURI : rl(null, "");
      break;
    default:
      t = t === 8 ? e.parentNode : e, e = t.namespaceURI || null, t = t.tagName, e = rl(e, t);
  }
  qt(An), Vt(An, e);
}
function lr() {
  qt(An), qt(ra), qt(aa);
}
function Wd(t) {
  Ci(aa.current);
  var e = Ci(An.current), n = rl(e, t.type);
  e !== n && (Vt(ra, t), Vt(An, n));
}
function wc(t) {
  ra.current === t && (qt(An), qt(ra));
}
var $t = ui(0);
function Eo(t) {
  for (var e = t; e !== null;) {
    if (e.tag === 13) {
      var n = e.memoizedState;
      if (n !== null && (n = n.dehydrated, n === null || n.data === "$?" || n.data === "$!")) return e;
    } else if (e.tag === 19 && e.memoizedProps.revealOrder !== void 0) {
      if (e.flags & 128) return e;
    } else if (e.child !== null) {
      e.child.return = e, e = e.child;
      continue;
    }
    if (e === t) break;
    for (; e.sibling === null;) {
      if (e.return === null || e.return === t) return null;
      e = e.return;
    }
    e.sibling.return = e.return, e = e.sibling;
  }
  return null;
}
var Ps = [];
function Cc() {
  for (var t = 0; t < Ps.length; t++) Ps[t]._workInProgressVersionPrimary = null;
  Ps.length = 0;
}
var eo = Xn.ReactCurrentDispatcher, Ms = Xn.ReactCurrentBatchConfig, Oi = 0, ee = null, le = null, he = null, ko = !1, Hr = !1, oa = 0, G1 = 0;
function ye() {
  throw Error(pt(321));
}
function Sc(t, e) {
  if (e === null) return !1;
  for (var n = 0; n < e.length && n < t.length; n++) if (!yn(t[n], e[n])) return !1;
  return !0;
}
function Ec(t, e, n, o, d, g) {
  if (Oi = g, ee = e, e.memoizedState = null, e.updateQueue = null, e.lanes = 0, eo.current = t === null || t.memoizedState === null ? K1 : J1, t = n(o, d), Hr) {
    g = 0;
    do {
      if (Hr = !1, oa = 0, 25 <= g) throw Error(pt(301));
      g += 1, he = le = null, e.updateQueue = null, eo.current = $1, t = n(o, d);
    } while (Hr);
  }
  if (eo.current = To, e = le !== null && le.next !== null, Oi = 0, he = le = ee = null, ko = !1, e) throw Error(pt(300));
  return t;
}
function kc() {
  var t = oa !== 0;
  return oa = 0, t;
}
function Sn() {
  var t = { memoizedState: null, baseState: null, baseQueue: null, queue: null, next: null };
  return he === null ? ee.memoizedState = he = t : he = he.next = t, he;
}
function an() {
  if (le === null) {
    var t = ee.alternate;
    t = t !== null ? t.memoizedState : null;
  } else t = le.next;
  var e = he === null ? ee.memoizedState : he.next;
  if (e !== null) he = e, le = t;
  else {
    if (t === null) throw Error(pt(310));
    le = t, t = { memoizedState: le.memoizedState, baseState: le.baseState, baseQueue: le.baseQueue, queue: le.queue, next: null }, he === null ? ee.memoizedState = he = t : he = he.next = t;
  }
  return he;
}
function sa(t, e) {
  return typeof e == "function" ? e(t) : e;
}
function As(t) {
  var e = an(), n = e.queue;
  if (n === null) throw Error(pt(311));
  n.lastRenderedReducer = t;
  var o = le, d = o.baseQueue, g = n.pending;
  if (g !== null) {
    if (d !== null) {
      var b = d.next;
      d.next = g.next, g.next = b;
    }
    o.baseQueue = d = g, n.pending = null;
  }
  if (d !== null) {
    g = d.next, o = o.baseState;
    var _ = b = null, w = null, E = g;
    do {
      var j = E.lane;
      if ((Oi & j) === j) w !== null && (w = w.next = { lane: 0, action: E.action, hasEagerState: E.hasEagerState, eagerState: E.eagerState, next: null }), o = E.hasEagerState ? E.eagerState : t(o, E.action);
      else {
        var H = {
          lane: j,
          action: E.action,
          hasEagerState: E.hasEagerState,
          eagerState: E.eagerState,
          next: null
        };
        w === null ? (_ = w = H, b = o) : w = w.next = H, ee.lanes |= j, Pi |= j;
      }
      E = E.next;
    } while (E !== null && E !== g);
    w === null ? b = o : w.next = _, yn(o, e.memoizedState) || (je = !0), e.memoizedState = o, e.baseState = b, e.baseQueue = w, n.lastRenderedState = o;
  }
  if (t = n.interleaved, t !== null) {
    d = t;
    do
      g = d.lane, ee.lanes |= g, Pi |= g, d = d.next;
    while (d !== t);
  } else d === null && (n.lanes = 0);
  return [e.memoizedState, n.dispatch];
}
function Ds(t) {
  var e = an(), n = e.queue;
  if (n === null) throw Error(pt(311));
  n.lastRenderedReducer = t;
  var o = n.dispatch, d = n.pending, g = e.memoizedState;
  if (d !== null) {
    n.pending = null;
    var b = d = d.next;
    do
      g = t(g, b.action), b = b.next;
    while (b !== d);
    yn(g, e.memoizedState) || (je = !0), e.memoizedState = g, e.baseQueue === null && (e.baseState = g), n.lastRenderedState = g;
  }
  return [g, o];
}
function Xd() {
}
function Yd(t, e) {
  var n = ee, o = an(), d = e(), g = !yn(o.memoizedState, d);
  if (g && (o.memoizedState = d, je = !0), o = o.queue, Tc(qd.bind(null, n, o, t), [t]), o.getSnapshot !== e || g || he !== null && he.memoizedState.tag & 1) {
    if (n.flags |= 2048, la(9, Gd.bind(null, n, o, d, e), void 0, null), fe === null) throw Error(pt(349));
    Oi & 30 || Vd(n, e, d);
  }
  return d;
}
function Vd(t, e, n) {
  t.flags |= 16384, t = { getSnapshot: e, value: n }, e = ee.updateQueue, e === null ? (e = { lastEffect: null, stores: null }, ee.updateQueue = e, e.stores = [t]) : (n = e.stores, n === null ? e.stores = [t] : n.push(t));
}
function Gd(t, e, n, o) {
  e.value = n, e.getSnapshot = o, Zd(e) && Qd(t);
}
function qd(t, e, n) {
  return n(function () {
    Zd(e) && Qd(t);
  });
}
function Zd(t) {
  var e = t.getSnapshot;
  t = t.value;
  try {
    var n = e();
    return !yn(t, n);
  } catch {
    return !0;
  }
}
function Qd(t) {
  var e = Hn(t, 1);
  e !== null && vn(e, t, 1, -1);
}
function hh(t) {
  var e = Sn();
  return typeof t == "function" && (t = t()), e.memoizedState = e.baseState = t, t = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: sa, lastRenderedState: t }, e.queue = t, t = t.dispatch = Q1.bind(null, ee, t), [e.memoizedState, t];
}
function la(t, e, n, o) {
  return t = { tag: t, create: e, destroy: n, deps: o, next: null }, e = ee.updateQueue, e === null ? (e = { lastEffect: null, stores: null }, ee.updateQueue = e, e.lastEffect = t.next = t) : (n = e.lastEffect, n === null ? e.lastEffect = t.next = t : (o = n.next, n.next = t, t.next = o, e.lastEffect = t)), t;
}
function Kd() {
  return an().memoizedState;
}
function no(t, e, n, o) {
  var d = Sn();
  ee.flags |= t, d.memoizedState = la(1 | e, n, void 0, o === void 0 ? null : o);
}
function Vo(t, e, n, o) {
  var d = an();
  o = o === void 0 ? null : o;
  var g = void 0;
  if (le !== null) {
    var b = le.memoizedState;
    if (g = b.destroy, o !== null && Sc(o, b.deps)) {
      d.memoizedState = la(e, n, g, o);
      return;
    }
  }
  ee.flags |= t, d.memoizedState = la(1 | e, n, g, o);
}
function fh(t, e) {
  return no(8390656, 8, t, e);
}
function Tc(t, e) {
  return Vo(2048, 8, t, e);
}
function Jd(t, e) {
  return Vo(4, 2, t, e);
}
function $d(t, e) {
  return Vo(4, 4, t, e);
}
function tp(t, e) {
  if (typeof e == "function") return t = t(), e(t), function () {
    e(null);
  };
  if (e != null) return t = t(), e.current = t, function () {
    e.current = null;
  };
}
function ep(t, e, n) {
  return n = n != null ? n.concat([t]) : null, Vo(4, 4, tp.bind(null, e, t), n);
}
function Oc() {
}
function np(t, e) {
  var n = an();
  e = e === void 0 ? null : e;
  var o = n.memoizedState;
  return o !== null && e !== null && Sc(e, o[1]) ? o[0] : (n.memoizedState = [t, e], t);
}
function ip(t, e) {
  var n = an();
  e = e === void 0 ? null : e;
  var o = n.memoizedState;
  return o !== null && e !== null && Sc(e, o[1]) ? o[0] : (t = t(), n.memoizedState = [t, e], t);
}
function rp(t, e, n) {
  return Oi & 21 ? (yn(n, e) || (n = cd(), ee.lanes |= n, Pi |= n, t.baseState = !0), e) : (t.baseState && (t.baseState = !1, je = !0), t.memoizedState = n);
}
function q1(t, e) {
  var n = Xt;
  Xt = n !== 0 && 4 > n ? n : 4, t(!0);
  var o = Ms.transition;
  Ms.transition = {};
  try {
    t(!1), e();
  } finally {
    Xt = n, Ms.transition = o;
  }
}
function ap() {
  return an().memoizedState;
}
function Z1(t, e, n) {
  var o = ai(t);
  if (n = { lane: o, action: n, hasEagerState: !1, eagerState: null, next: null }, op(t)) sp(e, n);
  else if (n = Ud(t, e, n, o), n !== null) {
    var d = Ee();
    vn(n, t, o, d), lp(n, e, o);
  }
}
function Q1(t, e, n) {
  var o = ai(t), d = { lane: o, action: n, hasEagerState: !1, eagerState: null, next: null };
  if (op(t)) sp(e, d);
  else {
    var g = t.alternate;
    if (t.lanes === 0 && (g === null || g.lanes === 0) && (g = e.lastRenderedReducer, g !== null)) try {
      var b = e.lastRenderedState, _ = g(b, n);
      if (d.hasEagerState = !0, d.eagerState = _, yn(_, b)) {
        var w = e.interleaved;
        w === null ? (d.next = d, bc(e)) : (d.next = w.next, w.next = d), e.interleaved = d;
        return;
      }
    } catch {
    } finally {
      }
    n = Ud(t, e, d, o), n !== null && (d = Ee(), vn(n, t, o, d), lp(n, e, o));
  }
}
function op(t) {
  var e = t.alternate;
  return t === ee || e !== null && e === ee;
}
function sp(t, e) {
  Hr = ko = !0;
  var n = t.pending;
  n === null ? e.next = e : (e.next = n.next, n.next = e), t.pending = e;
}
function lp(t, e, n) {
  if (n & 4194240) {
    var o = e.lanes;
    o &= t.pendingLanes, n |= o, e.lanes = n, ac(t, n);
  }
}
var To = { readContext: rn, useCallback: ye, useContext: ye, useEffect: ye, useImperativeHandle: ye, useInsertionEffect: ye, useLayoutEffect: ye, useMemo: ye, useReducer: ye, useRef: ye, useState: ye, useDebugValue: ye, useDeferredValue: ye, useTransition: ye, useMutableSource: ye, useSyncExternalStore: ye, useId: ye, unstable_isNewReconciler: !1 }, K1 = {
  readContext: rn, useCallback: function (t, e) {
    return Sn().memoizedState = [t, e === void 0 ? null : e], t;
  }, useContext: rn, useEffect: fh, useImperativeHandle: function (t, e, n) {
    return n = n != null ? n.concat([t]) : null, no(
      4194308,
      4,
      tp.bind(null, e, t),
      n
    );
  }, useLayoutEffect: function (t, e) {
    return no(4194308, 4, t, e);
  }, useInsertionEffect: function (t, e) {
    return no(4, 2, t, e);
  }, useMemo: function (t, e) {
    var n = Sn();
    return e = e === void 0 ? null : e, t = t(), n.memoizedState = [t, e], t;
  }, useReducer: function (t, e, n) {
    var o = Sn();
    return e = n !== void 0 ? n(e) : e, o.memoizedState = o.baseState = e, t = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: t, lastRenderedState: e }, o.queue = t, t = t.dispatch = Z1.bind(null, ee, t), [o.memoizedState, t];
  }, useRef: function (t) {
    var e = Sn();
    return t = { current: t }, e.memoizedState = t;
  }, useState: hh, useDebugValue: Oc, useDeferredValue: function (t) {
    return Sn().memoizedState = t;
  }, useTransition: function () {
    var t = hh(!1), e = t[0];
    return t = q1.bind(null, t[1]), Sn().memoizedState = t, [e, t];
  }, useMutableSource: function () {
  }, useSyncExternalStore: function (t, e, n) {
    var o = ee, d = Sn();
    if (Qt) {
      if (n === void 0) throw Error(pt(407));
      n = n();
    } else {
      if (n = e(), fe === null) throw Error(pt(349));
      Oi & 30 || Vd(o, e, n);
    }
    d.memoizedState = n;
    var g = { value: n, getSnapshot: e };
    return d.queue = g, fh(qd.bind(
      null,
      o,
      g,
      t
    ), [t]), o.flags |= 2048, la(9, Gd.bind(null, o, g, n, e), void 0, null), n;
  }, useId: function () {
    var t = Sn(), e = fe.identifierPrefix;
    if (Qt) {
      var n = Rn, o = In;
      n = (o & ~(1 << 32 - mn(o) - 1)).toString(32) + n, e = ":" + e + "R" + n, n = oa++, 0 < n && (e += "H" + n.toString(32)), e += ":";
    } else n = G1++, e = ":" + e + "r" + n.toString(32) + ":";
    return t.memoizedState = e;
  }, unstable_isNewReconciler: !1
}, J1 = {
  readContext: rn,
  useCallback: np,
  useContext: rn,
  useEffect: Tc,
  useImperativeHandle: ep,
  useInsertionEffect: Jd,
  useLayoutEffect: $d,
  useMemo: ip,
  useReducer: As,
  useRef: Kd,
  useState: function () {
    return As(sa);
  },
  useDebugValue: Oc,
  useDeferredValue: function (t) {
    var e = an();
    return rp(e, le.memoizedState, t);
  },
  useTransition: function () {
    var t = As(sa)[0], e = an().memoizedState;
    return [t, e];
  },
  useMutableSource: Xd,
  useSyncExternalStore: Yd,
  useId: ap,
  unstable_isNewReconciler: !1
}, $1 = {
  readContext: rn, useCallback: np, useContext: rn, useEffect: Tc, useImperativeHandle: ep, useInsertionEffect: Jd, useLayoutEffect: $d, useMemo: ip, useReducer: Ds, useRef: Kd, useState: function () {
    return Ds(sa);
  }, useDebugValue: Oc, useDeferredValue: function (t) {
    var e = an();
    return le === null ? e.memoizedState = t : rp(e, le.memoizedState, t);
  }, useTransition: function () {
    var t = Ds(sa)[0], e = an().memoizedState;
    return [t, e];
  }, useMutableSource: Xd, useSyncExternalStore: Yd, useId: ap, unstable_isNewReconciler: !1
};
function hn(t, e) {
  if (t && t.defaultProps) {
    e = ne({}, e), t = t.defaultProps;
    for (var n in t) e[n] === void 0 && (e[n] = t[n]);
    return e;
  }
  return e;
}
function El(t, e, n, o) {
  e = t.memoizedState, n = n(o, e), n = n == null ? e : ne({}, e, n), t.memoizedState = n, t.lanes === 0 && (t.updateQueue.baseState = n);
}
var Go = {
  isMounted: function (t) {
    return (t = t._reactInternals) ? Di(t) === t : !1;
  }, enqueueSetState: function (t, e, n) {
    t = t._reactInternals;
    var o = Ee(), d = ai(t), g = zn(o, d);
    g.payload = e, n != null && (g.callback = n), e = ii(t, g, d), e !== null && (vn(e, t, d, o), to(e, t, d));
  }, enqueueReplaceState: function (t, e, n) {
    t = t._reactInternals;
    var o = Ee(), d = ai(t), g = zn(o, d);
    g.tag = 1, g.payload = e, n != null && (g.callback = n), e = ii(t, g, d), e !== null && (vn(e, t, d, o), to(e, t, d));
  }, enqueueForceUpdate: function (t, e) {
    t = t._reactInternals;
    var n = Ee(), o = ai(t), d = zn(n, o);
    d.tag = 2, e != null && (d.callback = e), e = ii(t, d, o), e !== null && (vn(e, t, o, n), to(e, t, o));
  }
};
function dh(t, e, n, o, d, g, b) {
  return t = t.stateNode, typeof t.shouldComponentUpdate == "function" ? t.shouldComponentUpdate(o, g, b) : e.prototype && e.prototype.isPureReactComponent ? !ta(n, o) || !ta(d, g) : !0;
}
function cp(t, e, n) {
  var o = !1, d = li, g = e.contextType;
  return typeof g == "object" && g !== null ? g = rn(g) : (d = Fe(e) ? ki : _e.current, o = e.contextTypes, g = (o = o != null) ? ar(t, d) : li), e = new e(n, g), t.memoizedState = e.state !== null && e.state !== void 0 ? e.state : null, e.updater = Go, t.stateNode = e, e._reactInternals = t, o && (t = t.stateNode, t.__reactInternalMemoizedUnmaskedChildContext = d, t.__reactInternalMemoizedMaskedChildContext = g), e;
}
function ph(t, e, n, o) {
  t = e.state, typeof e.componentWillReceiveProps == "function" && e.componentWillReceiveProps(n, o), typeof e.UNSAFE_componentWillReceiveProps == "function" && e.UNSAFE_componentWillReceiveProps(n, o), e.state !== t && Go.enqueueReplaceState(e, e.state, null);
}
function kl(t, e, n, o) {
  var d = t.stateNode;
  d.props = n, d.state = t.memoizedState, d.refs = {}, xc(t);
  var g = e.contextType;
  typeof g == "object" && g !== null ? d.context = rn(g) : (g = Fe(e) ? ki : _e.current, d.context = ar(t, g)), d.state = t.memoizedState, g = e.getDerivedStateFromProps, typeof g == "function" && (El(t, e, g, n), d.state = t.memoizedState), typeof e.getDerivedStateFromProps == "function" || typeof d.getSnapshotBeforeUpdate == "function" || typeof d.UNSAFE_componentWillMount != "function" && typeof d.componentWillMount != "function" || (e = d.state, typeof d.componentWillMount == "function" && d.componentWillMount(), typeof d.UNSAFE_componentWillMount == "function" && d.UNSAFE_componentWillMount(), e !== d.state && Go.enqueueReplaceState(d, d.state, null), So(t, n, d, o), d.state = t.memoizedState), typeof d.componentDidMount == "function" && (t.flags |= 4194308);
}
function cr(t, e) {
  try {
    var n = "", o = e;
    do
      n += Tm(o), o = o.return;
    while (o);
    var d = n;
  } catch (g) {
    d = `
Error generating stack: ` + g.message + `
` + g.stack;
  }
  return { value: t, source: e, stack: d, digest: null };
}
function js(t, e, n) {
  return { value: t, source: null, stack: n ?? null, digest: e ?? null };
}
function Tl(t, e) {
  try {
    console.error(e.value);
  } catch (n) {
    setTimeout(function () {
      throw n;
    });
  }
}
var tv = typeof WeakMap == "function" ? WeakMap : Map;
function up(t, e, n) {
  n = zn(-1, n), n.tag = 3, n.payload = { element: null };
  var o = e.value;
  return n.callback = function () {
    Po || (Po = !0, Il = o), Tl(t, e);
  }, n;
}
function hp(t, e, n) {
  n = zn(-1, n), n.tag = 3;
  var o = t.type.getDerivedStateFromError;
  if (typeof o == "function") {
    var d = e.value;
    n.payload = function () {
      return o(d);
    }, n.callback = function () {
      Tl(t, e);
    };
  }
  var g = t.stateNode;
  return g !== null && typeof g.componentDidCatch == "function" && (n.callback = function () {
    Tl(t, e), typeof o != "function" && (ri === null ? ri = /* @__PURE__ */ new Set([this]) : ri.add(this));
    var b = e.stack;
    this.componentDidCatch(e.value, { componentStack: b !== null ? b : "" });
  }), n;
}
function gh(t, e, n) {
  var o = t.pingCache;
  if (o === null) {
    o = t.pingCache = new tv();
    var d = /* @__PURE__ */ new Set();
    o.set(e, d);
  } else d = o.get(e), d === void 0 && (d = /* @__PURE__ */ new Set(), o.set(e, d));
  d.has(n) || (d.add(n), t = pv.bind(null, t, e, n), e.then(t, t));
}
function mh(t) {
  do {
    var e;
    if ((e = t.tag === 13) && (e = t.memoizedState, e = e !== null ? e.dehydrated !== null : !0), e) return t;
    t = t.return;
  } while (t !== null);
  return null;
}
function vh(t, e, n, o, d) {
  return t.mode & 1 ? (t.flags |= 65536, t.lanes = d, t) : (t === e ? t.flags |= 65536 : (t.flags |= 128, n.flags |= 131072, n.flags &= -52805, n.tag === 1 && (n.alternate === null ? n.tag = 17 : (e = zn(-1, 1), e.tag = 2, ii(n, e, 1))), n.lanes |= 1), t);
}
var ev = Xn.ReactCurrentOwner, je = !1;
function Se(t, e, n, o) {
  e.child = t === null ? Nd(e, null, n, o) : sr(e, t.child, n, o);
}
function yh(t, e, n, o, d) {
  n = n.render;
  var g = e.ref;
  return nr(e, d), o = Ec(t, e, n, o, g, d), n = kc(), t !== null && !je ? (e.updateQueue = t.updateQueue, e.flags &= -2053, t.lanes &= ~d, Wn(t, e, d)) : (Qt && n && dc(e), e.flags |= 1, Se(t, e, o, d), e.child);
}
function bh(t, e, n, o, d) {
  if (t === null) {
    var g = n.type;
    return typeof g == "function" && !Ic(g) && g.defaultProps === void 0 && n.compare === null && n.defaultProps === void 0 ? (e.tag = 15, e.type = g, fp(t, e, g, o, d)) : (t = oo(n.type, null, o, e, e.mode, d), t.ref = e.ref, t.return = e, e.child = t);
  }
  if (g = t.child, !(t.lanes & d)) {
    var b = g.memoizedProps;
    if (n = n.compare, n = n !== null ? n : ta, n(b, o) && t.ref === e.ref) return Wn(t, e, d);
  }
  return e.flags |= 1, t = oi(g, o), t.ref = e.ref, t.return = e, e.child = t;
}
function fp(t, e, n, o, d) {
  if (t !== null) {
    var g = t.memoizedProps;
    if (ta(g, o) && t.ref === e.ref) if (je = !1, e.pendingProps = o = g, (t.lanes & d) !== 0) t.flags & 131072 && (je = !0);
    else return e.lanes = t.lanes, Wn(t, e, d);
  }
  return Ol(t, e, n, o, d);
}
function dp(t, e, n) {
  var o = e.pendingProps, d = o.children, g = t !== null ? t.memoizedState : null;
  if (o.mode === "hidden") if (!(e.mode & 1)) e.memoizedState = { baseLanes: 0, cachePool: null, transitions: null }, Vt(Qi, ze), ze |= n;
  else {
    if (!(n & 1073741824)) return t = g !== null ? g.baseLanes | n : n, e.lanes = e.childLanes = 1073741824, e.memoizedState = { baseLanes: t, cachePool: null, transitions: null }, e.updateQueue = null, Vt(Qi, ze), ze |= t, null;
    e.memoizedState = { baseLanes: 0, cachePool: null, transitions: null }, o = g !== null ? g.baseLanes : n, Vt(Qi, ze), ze |= o;
  }
  else g !== null ? (o = g.baseLanes | n, e.memoizedState = null) : o = n, Vt(Qi, ze), ze |= o;
  return Se(t, e, d, n), e.child;
}
function pp(t, e) {
  var n = e.ref;
  (t === null && n !== null || t !== null && t.ref !== n) && (e.flags |= 512, e.flags |= 2097152);
}
function Ol(t, e, n, o, d) {
  var g = Fe(n) ? ki : _e.current;
  return g = ar(e, g), nr(e, d), n = Ec(t, e, n, o, g, d), o = kc(), t !== null && !je ? (e.updateQueue = t.updateQueue, e.flags &= -2053, t.lanes &= ~d, Wn(t, e, d)) : (Qt && o && dc(e), e.flags |= 1, Se(t, e, n, d), e.child);
}
function xh(t, e, n, o, d) {
  if (Fe(n)) {
    var g = !0;
    bo(e);
  } else g = !1;
  if (nr(e, d), e.stateNode === null) io(t, e), cp(e, n, o), kl(e, n, o, d), o = !0;
  else if (t === null) {
    var b = e.stateNode, _ = e.memoizedProps;
    b.props = _;
    var w = b.context, E = n.contextType;
    typeof E == "object" && E !== null ? E = rn(E) : (E = Fe(n) ? ki : _e.current, E = ar(e, E));
    var j = n.getDerivedStateFromProps, H = typeof j == "function" || typeof b.getSnapshotBeforeUpdate == "function";
    H || typeof b.UNSAFE_componentWillReceiveProps != "function" && typeof b.componentWillReceiveProps != "function" || (_ !== o || w !== E) && ph(e, b, o, E), Gn = !1;
    var G = e.memoizedState;
    b.state = G, So(e, o, b, d), w = e.memoizedState, _ !== o || G !== w || Le.current || Gn ? (typeof j == "function" && (El(e, n, j, o), w = e.memoizedState), (_ = Gn || dh(e, n, _, o, G, w, E)) ? (H || typeof b.UNSAFE_componentWillMount != "function" && typeof b.componentWillMount != "function" || (typeof b.componentWillMount == "function" && b.componentWillMount(), typeof b.UNSAFE_componentWillMount == "function" && b.UNSAFE_componentWillMount()), typeof b.componentDidMount == "function" && (e.flags |= 4194308)) : (typeof b.componentDidMount == "function" && (e.flags |= 4194308), e.memoizedProps = o, e.memoizedState = w), b.props = o, b.state = w, b.context = E, o = _) : (typeof b.componentDidMount == "function" && (e.flags |= 4194308), o = !1);
  } else {
    b = e.stateNode, Hd(t, e), _ = e.memoizedProps, E = e.type === e.elementType ? _ : hn(e.type, _), b.props = E, H = e.pendingProps, G = b.context, w = n.contextType, typeof w == "object" && w !== null ? w = rn(w) : (w = Fe(n) ? ki : _e.current, w = ar(e, w));
    var J = n.getDerivedStateFromProps;
    (j = typeof J == "function" || typeof b.getSnapshotBeforeUpdate == "function") || typeof b.UNSAFE_componentWillReceiveProps != "function" && typeof b.componentWillReceiveProps != "function" || (_ !== H || G !== w) && ph(e, b, o, w), Gn = !1, G = e.memoizedState, b.state = G, So(e, o, b, d);
    var nt = e.memoizedState;
    _ !== H || G !== nt || Le.current || Gn ? (typeof J == "function" && (El(e, n, J, o), nt = e.memoizedState), (E = Gn || dh(e, n, E, o, G, nt, w) || !1) ? (j || typeof b.UNSAFE_componentWillUpdate != "function" && typeof b.componentWillUpdate != "function" || (typeof b.componentWillUpdate == "function" && b.componentWillUpdate(o, nt, w), typeof b.UNSAFE_componentWillUpdate == "function" && b.UNSAFE_componentWillUpdate(o, nt, w)), typeof b.componentDidUpdate == "function" && (e.flags |= 4), typeof b.getSnapshotBeforeUpdate == "function" && (e.flags |= 1024)) : (typeof b.componentDidUpdate != "function" || _ === t.memoizedProps && G === t.memoizedState || (e.flags |= 4), typeof b.getSnapshotBeforeUpdate != "function" || _ === t.memoizedProps && G === t.memoizedState || (e.flags |= 1024), e.memoizedProps = o, e.memoizedState = nt), b.props = o, b.state = nt, b.context = w, o = E) : (typeof b.componentDidUpdate != "function" || _ === t.memoizedProps && G === t.memoizedState || (e.flags |= 4), typeof b.getSnapshotBeforeUpdate != "function" || _ === t.memoizedProps && G === t.memoizedState || (e.flags |= 1024), o = !1);
  }
  return Pl(t, e, n, o, g, d);
}
function Pl(t, e, n, o, d, g) {
  pp(t, e);
  var b = (e.flags & 128) !== 0;
  if (!o && !b) return d && ah(e, n, !1), Wn(t, e, g);
  o = e.stateNode, ev.current = e;
  var _ = b && typeof n.getDerivedStateFromError != "function" ? null : o.render();
  return e.flags |= 1, t !== null && b ? (e.child = sr(e, t.child, null, g), e.child = sr(e, null, _, g)) : Se(t, e, _, g), e.memoizedState = o.state, d && ah(e, n, !0), e.child;
}
function gp(t) {
  var e = t.stateNode;
  e.pendingContext ? rh(t, e.pendingContext, e.pendingContext !== e.context) : e.context && rh(t, e.context, !1), _c(t, e.containerInfo);
}
function _h(t, e, n, o, d) {
  return or(), gc(d), e.flags |= 256, Se(t, e, n, o), e.child;
}
var Ml = { dehydrated: null, treeContext: null, retryLane: 0 };
function Al(t) {
  return { baseLanes: t, cachePool: null, transitions: null };
}
function mp(t, e, n) {
  var o = e.pendingProps, d = $t.current, g = !1, b = (e.flags & 128) !== 0, _;
  if ((_ = b) || (_ = t !== null && t.memoizedState === null ? !1 : (d & 2) !== 0), _ ? (g = !0, e.flags &= -129) : (t === null || t.memoizedState !== null) && (d |= 1), Vt($t, d & 1), t === null)
    return Cl(e), t = e.memoizedState, t !== null && (t = t.dehydrated, t !== null) ? (e.mode & 1 ? t.data === "$!" ? e.lanes = 8 : e.lanes = 1073741824 : e.lanes = 1, null) : (b = o.children, t = o.fallback, g ? (o = e.mode, g = e.child, b = { mode: "hidden", children: b }, !(o & 1) && g !== null ? (g.childLanes = 0, g.pendingProps = b) : g = Qo(b, o, 0, null), t = Ei(t, o, n, null), g.return = e, t.return = e, g.sibling = t, e.child = g, e.child.memoizedState = Al(n), e.memoizedState = Ml, t) : Pc(e, b));
  if (d = t.memoizedState, d !== null && (_ = d.dehydrated, _ !== null)) return nv(t, e, b, o, _, d, n);
  if (g) {
    g = o.fallback, b = e.mode, d = t.child, _ = d.sibling;
    var w = { mode: "hidden", children: o.children };
    return !(b & 1) && e.child !== d ? (o = e.child, o.childLanes = 0, o.pendingProps = w, e.deletions = null) : (o = oi(d, w), o.subtreeFlags = d.subtreeFlags & 14680064), _ !== null ? g = oi(_, g) : (g = Ei(g, b, n, null), g.flags |= 2), g.return = e, o.return = e, o.sibling = g, e.child = o, o = g, g = e.child, b = t.child.memoizedState, b = b === null ? Al(n) : { baseLanes: b.baseLanes | n, cachePool: null, transitions: b.transitions }, g.memoizedState = b, g.childLanes = t.childLanes & ~n, e.memoizedState = Ml, o;
  }
  return g = t.child, t = g.sibling, o = oi(g, { mode: "visible", children: o.children }), !(e.mode & 1) && (o.lanes = n), o.return = e, o.sibling = null, t !== null && (n = e.deletions, n === null ? (e.deletions = [t], e.flags |= 16) : n.push(t)), e.child = o, e.memoizedState = null, o;
}
function Pc(t, e) {
  return e = Qo({ mode: "visible", children: e }, t.mode, 0, null), e.return = t, t.child = e;
}
function Ba(t, e, n, o) {
  return o !== null && gc(o), sr(e, t.child, null, n), t = Pc(e, e.pendingProps.children), t.flags |= 2, e.memoizedState = null, t;
}
function nv(t, e, n, o, d, g, b) {
  if (n)
    return e.flags & 256 ? (e.flags &= -257, o = js(Error(pt(422))), Ba(t, e, b, o)) : e.memoizedState !== null ? (e.child = t.child, e.flags |= 128, null) : (g = o.fallback, d = e.mode, o = Qo({ mode: "visible", children: o.children }, d, 0, null), g = Ei(g, d, b, null), g.flags |= 2, o.return = e, g.return = e, o.sibling = g, e.child = o, e.mode & 1 && sr(e, t.child, null, b), e.child.memoizedState = Al(b), e.memoizedState = Ml, g);
  if (!(e.mode & 1)) return Ba(t, e, b, null);
  if (d.data === "$!") {
    if (o = d.nextSibling && d.nextSibling.dataset, o) var _ = o.dgst;
    return o = _, g = Error(pt(419)), o = js(g, o, void 0), Ba(t, e, b, o);
  }
  if (_ = (b & t.childLanes) !== 0, je || _) {
    if (o = fe, o !== null) {
      switch (b & -b) {
        case 4:
          d = 2;
          break;
        case 16:
          d = 8;
          break;
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
        case 16384:
        case 32768:
        case 65536:
        case 131072:
        case 262144:
        case 524288:
        case 1048576:
        case 2097152:
        case 4194304:
        case 8388608:
        case 16777216:
        case 33554432:
        case 67108864:
          d = 32;
          break;
        case 536870912:
          d = 268435456;
          break;
        default:
          d = 0;
      }
      d = d & (o.suspendedLanes | b) ? 0 : d, d !== 0 && d !== g.retryLane && (g.retryLane = d, Hn(t, d), vn(o, t, d, -1));
    }
    return Fc(), o = js(Error(pt(421))), Ba(t, e, b, o);
  }
  return d.data === "$?" ? (e.flags |= 128, e.child = t.child, e = gv.bind(null, t), d._reactRetry = e, null) : (t = g.treeContext, Ne = ni(d.nextSibling), Ue = e, Qt = !0, dn = null, t !== null && (Qe[Ke++] = In, Qe[Ke++] = Rn, Qe[Ke++] = Ti, In = t.id, Rn = t.overflow, Ti = e), e = Pc(e, o.children), e.flags |= 4096, e);
}
function wh(t, e, n) {
  t.lanes |= e;
  var o = t.alternate;
  o !== null && (o.lanes |= e), Sl(t.return, e, n);
}
function Ls(t, e, n, o, d) {
  var g = t.memoizedState;
  g === null ? t.memoizedState = { isBackwards: e, rendering: null, renderingStartTime: 0, last: o, tail: n, tailMode: d } : (g.isBackwards = e, g.rendering = null, g.renderingStartTime = 0, g.last = o, g.tail = n, g.tailMode = d);
}
function vp(t, e, n) {
  var o = e.pendingProps, d = o.revealOrder, g = o.tail;
  if (Se(t, e, o.children, n), o = $t.current, o & 2) o = o & 1 | 2, e.flags |= 128;
  else {
    if (t !== null && t.flags & 128) t: for (t = e.child; t !== null;) {
      if (t.tag === 13) t.memoizedState !== null && wh(t, n, e);
      else if (t.tag === 19) wh(t, n, e);
      else if (t.child !== null) {
        t.child.return = t, t = t.child;
        continue;
      }
      if (t === e) break t;
      for (; t.sibling === null;) {
        if (t.return === null || t.return === e) break t;
        t = t.return;
      }
      t.sibling.return = t.return, t = t.sibling;
    }
    o &= 1;
  }
  if (Vt($t, o), !(e.mode & 1)) e.memoizedState = null;
  else switch (d) {
    case "forwards":
      for (n = e.child, d = null; n !== null;) t = n.alternate, t !== null && Eo(t) === null && (d = n), n = n.sibling;
      n = d, n === null ? (d = e.child, e.child = null) : (d = n.sibling, n.sibling = null), Ls(e, !1, d, n, g);
      break;
    case "backwards":
      for (n = null, d = e.child, e.child = null; d !== null;) {
        if (t = d.alternate, t !== null && Eo(t) === null) {
          e.child = d;
          break;
        }
        t = d.sibling, d.sibling = n, n = d, d = t;
      }
      Ls(e, !0, n, null, g);
      break;
    case "together":
      Ls(e, !1, null, null, void 0);
      break;
    default:
      e.memoizedState = null;
  }
  return e.child;
}
function io(t, e) {
  !(e.mode & 1) && t !== null && (t.alternate = null, e.alternate = null, e.flags |= 2);
}
function Wn(t, e, n) {
  if (t !== null && (e.dependencies = t.dependencies), Pi |= e.lanes, !(n & e.childLanes)) return null;
  if (t !== null && e.child !== t.child) throw Error(pt(153));
  if (e.child !== null) {
    for (t = e.child, n = oi(t, t.pendingProps), e.child = n, n.return = e; t.sibling !== null;) t = t.sibling, n = n.sibling = oi(t, t.pendingProps), n.return = e;
    n.sibling = null;
  }
  return e.child;
}
function iv(t, e, n) {
  switch (e.tag) {
    case 3:
      gp(e), or();
      break;
    case 5:
      Wd(e);
      break;
    case 1:
      Fe(e.type) && bo(e);
      break;
    case 4:
      _c(e, e.stateNode.containerInfo);
      break;
    case 10:
      var o = e.type._context, d = e.memoizedProps.value;
      Vt(wo, o._currentValue), o._currentValue = d;
      break;
    case 13:
      if (o = e.memoizedState, o !== null)
        return o.dehydrated !== null ? (Vt($t, $t.current & 1), e.flags |= 128, null) : n & e.child.childLanes ? mp(t, e, n) : (Vt($t, $t.current & 1), t = Wn(t, e, n), t !== null ? t.sibling : null);
      Vt($t, $t.current & 1);
      break;
    case 19:
      if (o = (n & e.childLanes) !== 0, t.flags & 128) {
        if (o) return vp(t, e, n);
        e.flags |= 128;
      }
      if (d = e.memoizedState, d !== null && (d.rendering = null, d.tail = null, d.lastEffect = null), Vt($t, $t.current), o) break;
      return null;
    case 22:
    case 23:
      return e.lanes = 0, dp(t, e, n);
  }
  return Wn(t, e, n);
}
var yp, Dl, bp, xp;
yp = function (t, e) {
  for (var n = e.child; n !== null;) {
    if (n.tag === 5 || n.tag === 6) t.appendChild(n.stateNode);
    else if (n.tag !== 4 && n.child !== null) {
      n.child.return = n, n = n.child;
      continue;
    }
    if (n === e) break;
    for (; n.sibling === null;) {
      if (n.return === null || n.return === e) return;
      n = n.return;
    }
    n.sibling.return = n.return, n = n.sibling;
  }
};
Dl = function () {
};
bp = function (t, e, n, o) {
  var d = t.memoizedProps;
  if (d !== o) {
    t = e.stateNode, Ci(An.current);
    var g = null;
    switch (n) {
      case "input":
        d = tl(t, d), o = tl(t, o), g = [];
        break;
      case "select":
        d = ne({}, d, { value: void 0 }), o = ne({}, o, { value: void 0 }), g = [];
        break;
      case "textarea":
        d = il(t, d), o = il(t, o), g = [];
        break;
      default:
        typeof d.onClick != "function" && typeof o.onClick == "function" && (t.onclick = vo);
    }
    al(n, o);
    var b;
    n = null;
    for (E in d) if (!o.hasOwnProperty(E) && d.hasOwnProperty(E) && d[E] != null) if (E === "style") {
      var _ = d[E];
      for (b in _) _.hasOwnProperty(b) && (n || (n = {}), n[b] = "");
    } else E !== "dangerouslySetInnerHTML" && E !== "children" && E !== "suppressContentEditableWarning" && E !== "suppressHydrationWarning" && E !== "autoFocus" && (Gr.hasOwnProperty(E) ? g || (g = []) : (g = g || []).push(E, null));
    for (E in o) {
      var w = o[E];
      if (_ = d != null ? d[E] : void 0, o.hasOwnProperty(E) && w !== _ && (w != null || _ != null)) if (E === "style") if (_) {
        for (b in _) !_.hasOwnProperty(b) || w && w.hasOwnProperty(b) || (n || (n = {}), n[b] = "");
        for (b in w) w.hasOwnProperty(b) && _[b] !== w[b] && (n || (n = {}), n[b] = w[b]);
      } else n || (g || (g = []), g.push(
        E,
        n
      )), n = w;
      else E === "dangerouslySetInnerHTML" ? (w = w ? w.__html : void 0, _ = _ ? _.__html : void 0, w != null && _ !== w && (g = g || []).push(E, w)) : E === "children" ? typeof w != "string" && typeof w != "number" || (g = g || []).push(E, "" + w) : E !== "suppressContentEditableWarning" && E !== "suppressHydrationWarning" && (Gr.hasOwnProperty(E) ? (w != null && E === "onScroll" && Gt("scroll", t), g || _ === w || (g = [])) : (g = g || []).push(E, w));
    }
    n && (g = g || []).push("style", n);
    var E = g;
    (e.updateQueue = E) && (e.flags |= 4);
  }
};
xp = function (t, e, n, o) {
  n !== o && (e.flags |= 4);
};
function Ar(t, e) {
  if (!Qt) switch (t.tailMode) {
    case "hidden":
      e = t.tail;
      for (var n = null; e !== null;) e.alternate !== null && (n = e), e = e.sibling;
      n === null ? t.tail = null : n.sibling = null;
      break;
    case "collapsed":
      n = t.tail;
      for (var o = null; n !== null;) n.alternate !== null && (o = n), n = n.sibling;
      o === null ? e || t.tail === null ? t.tail = null : t.tail.sibling = null : o.sibling = null;
  }
}
function be(t) {
  var e = t.alternate !== null && t.alternate.child === t.child, n = 0, o = 0;
  if (e) for (var d = t.child; d !== null;) n |= d.lanes | d.childLanes, o |= d.subtreeFlags & 14680064, o |= d.flags & 14680064, d.return = t, d = d.sibling;
  else for (d = t.child; d !== null;) n |= d.lanes | d.childLanes, o |= d.subtreeFlags, o |= d.flags, d.return = t, d = d.sibling;
  return t.subtreeFlags |= o, t.childLanes = n, e;
}
function rv(t, e, n) {
  var o = e.pendingProps;
  switch (pc(e), e.tag) {
    case 2:
    case 16:
    case 15:
    case 0:
    case 11:
    case 7:
    case 8:
    case 12:
    case 9:
    case 14:
      return be(e), null;
    case 1:
      return Fe(e.type) && yo(), be(e), null;
    case 3:
      return o = e.stateNode, lr(), qt(Le), qt(_e), Cc(), o.pendingContext && (o.context = o.pendingContext, o.pendingContext = null), (t === null || t.child === null) && (Ia(e) ? e.flags |= 4 : t === null || t.memoizedState.isDehydrated && !(e.flags & 256) || (e.flags |= 1024, dn !== null && (zl(dn), dn = null))), Dl(t, e), be(e), null;
    case 5:
      wc(e);
      var d = Ci(aa.current);
      if (n = e.type, t !== null && e.stateNode != null) bp(t, e, n, o, d), t.ref !== e.ref && (e.flags |= 512, e.flags |= 2097152);
      else {
        if (!o) {
          if (e.stateNode === null) throw Error(pt(166));
          return be(e), null;
        }
        if (t = Ci(An.current), Ia(e)) {
          o = e.stateNode, n = e.type;
          var g = e.memoizedProps;
          switch (o[En] = e, o[ia] = g, t = (e.mode & 1) !== 0, n) {
            case "dialog":
              Gt("cancel", o), Gt("close", o);
              break;
            case "iframe":
            case "object":
            case "embed":
              Gt("load", o);
              break;
            case "video":
            case "audio":
              for (d = 0; d < Ir.length; d++) Gt(Ir[d], o);
              break;
            case "source":
              Gt("error", o);
              break;
            case "img":
            case "image":
            case "link":
              Gt(
                "error",
                o
              ), Gt("load", o);
              break;
            case "details":
              Gt("toggle", o);
              break;
            case "input":
              Au(o, g), Gt("invalid", o);
              break;
            case "select":
              o._wrapperState = { wasMultiple: !!g.multiple }, Gt("invalid", o);
              break;
            case "textarea":
              ju(o, g), Gt("invalid", o);
          }
          al(n, g), d = null;
          for (var b in g) if (g.hasOwnProperty(b)) {
            var _ = g[b];
            b === "children" ? typeof _ == "string" ? o.textContent !== _ && (g.suppressHydrationWarning !== !0 && Fa(o.textContent, _, t), d = ["children", _]) : typeof _ == "number" && o.textContent !== "" + _ && (g.suppressHydrationWarning !== !0 && Fa(
              o.textContent,
              _,
              t
            ), d = ["children", "" + _]) : Gr.hasOwnProperty(b) && _ != null && b === "onScroll" && Gt("scroll", o);
          }
          switch (n) {
            case "input":
              Ta(o), Du(o, g, !0);
              break;
            case "textarea":
              Ta(o), Lu(o);
              break;
            case "select":
            case "option":
              break;
            default:
              typeof g.onClick == "function" && (o.onclick = vo);
          }
          o = d, e.updateQueue = o, o !== null && (e.flags |= 4);
        } else {
          b = d.nodeType === 9 ? d : d.ownerDocument, t === "http://www.w3.org/1999/xhtml" && (t = qf(n)), t === "http://www.w3.org/1999/xhtml" ? n === "script" ? (t = b.createElement("div"), t.innerHTML = "<script><\/script>", t = t.removeChild(t.firstChild)) : typeof o.is == "string" ? t = b.createElement(n, { is: o.is }) : (t = b.createElement(n), n === "select" && (b = t, o.multiple ? b.multiple = !0 : o.size && (b.size = o.size))) : t = b.createElementNS(t, n), t[En] = e, t[ia] = o, yp(t, e, !1, !1), e.stateNode = t;
          t: {
            switch (b = ol(n, o), n) {
              case "dialog":
                Gt("cancel", t), Gt("close", t), d = o;
                break;
              case "iframe":
              case "object":
              case "embed":
                Gt("load", t), d = o;
                break;
              case "video":
              case "audio":
                for (d = 0; d < Ir.length; d++) Gt(Ir[d], t);
                d = o;
                break;
              case "source":
                Gt("error", t), d = o;
                break;
              case "img":
              case "image":
              case "link":
                Gt(
                  "error",
                  t
                ), Gt("load", t), d = o;
                break;
              case "details":
                Gt("toggle", t), d = o;
                break;
              case "input":
                Au(t, o), d = tl(t, o), Gt("invalid", t);
                break;
              case "option":
                d = o;
                break;
              case "select":
                t._wrapperState = { wasMultiple: !!o.multiple }, d = ne({}, o, { value: void 0 }), Gt("invalid", t);
                break;
              case "textarea":
                ju(t, o), d = il(t, o), Gt("invalid", t);
                break;
              default:
                d = o;
            }
            al(n, d), _ = d;
            for (g in _) if (_.hasOwnProperty(g)) {
              var w = _[g];
              g === "style" ? Kf(t, w) : g === "dangerouslySetInnerHTML" ? (w = w ? w.__html : void 0, w != null && Zf(t, w)) : g === "children" ? typeof w == "string" ? (n !== "textarea" || w !== "") && qr(t, w) : typeof w == "number" && qr(t, "" + w) : g !== "suppressContentEditableWarning" && g !== "suppressHydrationWarning" && g !== "autoFocus" && (Gr.hasOwnProperty(g) ? w != null && g === "onScroll" && Gt("scroll", t) : w != null && $l(t, g, w, b));
            }
            switch (n) {
              case "input":
                Ta(t), Du(t, o, !1);
                break;
              case "textarea":
                Ta(t), Lu(t);
                break;
              case "option":
                o.value != null && t.setAttribute("value", "" + si(o.value));
                break;
              case "select":
                t.multiple = !!o.multiple, g = o.value, g != null ? Ji(t, !!o.multiple, g, !1) : o.defaultValue != null && Ji(
                  t,
                  !!o.multiple,
                  o.defaultValue,
                  !0
                );
                break;
              default:
                typeof d.onClick == "function" && (t.onclick = vo);
            }
            switch (n) {
              case "button":
              case "input":
              case "select":
              case "textarea":
                o = !!o.autoFocus;
                break t;
              case "img":
                o = !0;
                break t;
              default:
                o = !1;
            }
          }
          o && (e.flags |= 4);
        }
        e.ref !== null && (e.flags |= 512, e.flags |= 2097152);
      }
      return be(e), null;
    case 6:
      if (t && e.stateNode != null) xp(t, e, t.memoizedProps, o);
      else {
        if (typeof o != "string" && e.stateNode === null) throw Error(pt(166));
        if (n = Ci(aa.current), Ci(An.current), Ia(e)) {
          if (o = e.stateNode, n = e.memoizedProps, o[En] = e, (g = o.nodeValue !== n) && (t = Ue, t !== null)) switch (t.tag) {
            case 3:
              Fa(o.nodeValue, n, (t.mode & 1) !== 0);
              break;
            case 5:
              t.memoizedProps.suppressHydrationWarning !== !0 && Fa(o.nodeValue, n, (t.mode & 1) !== 0);
          }
          g && (e.flags |= 4);
        } else o = (n.nodeType === 9 ? n : n.ownerDocument).createTextNode(o), o[En] = e, e.stateNode = o;
      }
      return be(e), null;
    case 13:
      if (qt($t), o = e.memoizedState, t === null || t.memoizedState !== null && t.memoizedState.dehydrated !== null) {
        if (Qt && Ne !== null && e.mode & 1 && !(e.flags & 128)) Bd(), or(), e.flags |= 98560, g = !1;
        else if (g = Ia(e), o !== null && o.dehydrated !== null) {
          if (t === null) {
            if (!g) throw Error(pt(318));
            if (g = e.memoizedState, g = g !== null ? g.dehydrated : null, !g) throw Error(pt(317));
            g[En] = e;
          } else or(), !(e.flags & 128) && (e.memoizedState = null), e.flags |= 4;
          be(e), g = !1;
        } else dn !== null && (zl(dn), dn = null), g = !0;
        if (!g) return e.flags & 65536 ? e : null;
      }
      return e.flags & 128 ? (e.lanes = n, e) : (o = o !== null, o !== (t !== null && t.memoizedState !== null) && o && (e.child.flags |= 8192, e.mode & 1 && (t === null || $t.current & 1 ? ce === 0 && (ce = 3) : Fc())), e.updateQueue !== null && (e.flags |= 4), be(e), null);
    case 4:
      return lr(), Dl(t, e), t === null && ea(e.stateNode.containerInfo), be(e), null;
    case 10:
      return yc(e.type._context), be(e), null;
    case 17:
      return Fe(e.type) && yo(), be(e), null;
    case 19:
      if (qt($t), g = e.memoizedState, g === null) return be(e), null;
      if (o = (e.flags & 128) !== 0, b = g.rendering, b === null) if (o) Ar(g, !1);
      else {
        if (ce !== 0 || t !== null && t.flags & 128) for (t = e.child; t !== null;) {
          if (b = Eo(t), b !== null) {
            for (e.flags |= 128, Ar(g, !1), o = b.updateQueue, o !== null && (e.updateQueue = o, e.flags |= 4), e.subtreeFlags = 0, o = n, n = e.child; n !== null;) g = n, t = o, g.flags &= 14680066, b = g.alternate, b === null ? (g.childLanes = 0, g.lanes = t, g.child = null, g.subtreeFlags = 0, g.memoizedProps = null, g.memoizedState = null, g.updateQueue = null, g.dependencies = null, g.stateNode = null) : (g.childLanes = b.childLanes, g.lanes = b.lanes, g.child = b.child, g.subtreeFlags = 0, g.deletions = null, g.memoizedProps = b.memoizedProps, g.memoizedState = b.memoizedState, g.updateQueue = b.updateQueue, g.type = b.type, t = b.dependencies, g.dependencies = t === null ? null : { lanes: t.lanes, firstContext: t.firstContext }), n = n.sibling;
            return Vt($t, $t.current & 1 | 2), e.child;
          }
          t = t.sibling;
        }
        g.tail !== null && ae() > ur && (e.flags |= 128, o = !0, Ar(g, !1), e.lanes = 4194304);
      }
      else {
        if (!o) if (t = Eo(b), t !== null) {
          if (e.flags |= 128, o = !0, n = t.updateQueue, n !== null && (e.updateQueue = n, e.flags |= 4), Ar(g, !0), g.tail === null && g.tailMode === "hidden" && !b.alternate && !Qt) return be(e), null;
        } else 2 * ae() - g.renderingStartTime > ur && n !== 1073741824 && (e.flags |= 128, o = !0, Ar(g, !1), e.lanes = 4194304);
        g.isBackwards ? (b.sibling = e.child, e.child = b) : (n = g.last, n !== null ? n.sibling = b : e.child = b, g.last = b);
      }
      return g.tail !== null ? (e = g.tail, g.rendering = e, g.tail = e.sibling, g.renderingStartTime = ae(), e.sibling = null, n = $t.current, Vt($t, o ? n & 1 | 2 : n & 1), e) : (be(e), null);
    case 22:
    case 23:
      return Lc(), o = e.memoizedState !== null, t !== null && t.memoizedState !== null !== o && (e.flags |= 8192), o && e.mode & 1 ? ze & 1073741824 && (be(e), e.subtreeFlags & 6 && (e.flags |= 8192)) : be(e), null;
    case 24:
      return null;
    case 25:
      return null;
  }
  throw Error(pt(156, e.tag));
}
function av(t, e) {
  switch (pc(e), e.tag) {
    case 1:
      return Fe(e.type) && yo(), t = e.flags, t & 65536 ? (e.flags = t & -65537 | 128, e) : null;
    case 3:
      return lr(), qt(Le), qt(_e), Cc(), t = e.flags, t & 65536 && !(t & 128) ? (e.flags = t & -65537 | 128, e) : null;
    case 5:
      return wc(e), null;
    case 13:
      if (qt($t), t = e.memoizedState, t !== null && t.dehydrated !== null) {
        if (e.alternate === null) throw Error(pt(340));
        or();
      }
      return t = e.flags, t & 65536 ? (e.flags = t & -65537 | 128, e) : null;
    case 19:
      return qt($t), null;
    case 4:
      return lr(), null;
    case 10:
      return yc(e.type._context), null;
    case 22:
    case 23:
      return Lc(), null;
    case 24:
      return null;
    default:
      return null;
  }
}
var za = !1, xe = !1, ov = typeof WeakSet == "function" ? WeakSet : Set, xt = null;
function Zi(t, e) {
  var n = t.ref;
  if (n !== null) if (typeof n == "function") try {
    n(null);
  } catch (o) {
    re(t, e, o);
  }
  else n.current = null;
}
function _p(t, e, n) {
  try {
    n();
  } catch (o) {
    re(t, e, o);
  }
}
var Ch = !1;
function sv(t, e) {
  if (ml = po, t = Ed(), fc(t)) {
    if ("selectionStart" in t) var n = { start: t.selectionStart, end: t.selectionEnd };
    else t: {
      n = (n = t.ownerDocument) && n.defaultView || window;
      var o = n.getSelection && n.getSelection();
      if (o && o.rangeCount !== 0) {
        n = o.anchorNode;
        var d = o.anchorOffset, g = o.focusNode;
        o = o.focusOffset;
        try {
          n.nodeType, g.nodeType;
        } catch {
          n = null;
          break t;
        }
        var b = 0, _ = -1, w = -1, E = 0, j = 0, H = t, G = null;
        e: for (; ;) {
          for (var J; H !== n || d !== 0 && H.nodeType !== 3 || (_ = b + d), H !== g || o !== 0 && H.nodeType !== 3 || (w = b + o), H.nodeType === 3 && (b += H.nodeValue.length), (J = H.firstChild) !== null;)
            G = H, H = J;
          for (; ;) {
            if (H === t) break e;
            if (G === n && ++E === d && (_ = b), G === g && ++j === o && (w = b), (J = H.nextSibling) !== null) break;
            H = G, G = H.parentNode;
          }
          H = J;
        }
        n = _ === -1 || w === -1 ? null : { start: _, end: w };
      } else n = null;
    }
    n = n || { start: 0, end: 0 };
  } else n = null;
  for (vl = { focusedElem: t, selectionRange: n }, po = !1, xt = e; xt !== null;) if (e = xt, t = e.child, (e.subtreeFlags & 1028) !== 0 && t !== null) t.return = e, xt = t;
  else for (; xt !== null;) {
    e = xt;
    try {
      var nt = e.alternate;
      if (e.flags & 1024) switch (e.tag) {
        case 0:
        case 11:
        case 15:
          break;
        case 1:
          if (nt !== null) {
            var rt = nt.memoizedProps, vt = nt.memoizedState, Y = e.stateNode, L = Y.getSnapshotBeforeUpdate(e.elementType === e.type ? rt : hn(e.type, rt), vt);
            Y.__reactInternalSnapshotBeforeUpdate = L;
          }
          break;
        case 3:
          var A = e.stateNode.containerInfo;
          A.nodeType === 1 ? A.textContent = "" : A.nodeType === 9 && A.documentElement && A.removeChild(A.documentElement);
          break;
        case 5:
        case 6:
        case 4:
        case 17:
          break;
        default:
          throw Error(pt(163));
      }
    } catch (tt) {
      re(e, e.return, tt);
    }
    if (t = e.sibling, t !== null) {
      t.return = e.return, xt = t;
      break;
    }
    xt = e.return;
  }
  return nt = Ch, Ch = !1, nt;
}
function Wr(t, e, n) {
  var o = e.updateQueue;
  if (o = o !== null ? o.lastEffect : null, o !== null) {
    var d = o = o.next;
    do {
      if ((d.tag & t) === t) {
        var g = d.destroy;
        d.destroy = void 0, g !== void 0 && _p(e, n, g);
      }
      d = d.next;
    } while (d !== o);
  }
}
function qo(t, e) {
  if (e = e.updateQueue, e = e !== null ? e.lastEffect : null, e !== null) {
    var n = e = e.next;
    do {
      if ((n.tag & t) === t) {
        var o = n.create;
        n.destroy = o();
      }
      n = n.next;
    } while (n !== e);
  }
}
function jl(t) {
  var e = t.ref;
  if (e !== null) {
    var n = t.stateNode;
    switch (t.tag) {
      case 5:
        t = n;
        break;
      default:
        t = n;
    }
    typeof e == "function" ? e(t) : e.current = t;
  }
}
function wp(t) {
  var e = t.alternate;
  e !== null && (t.alternate = null, wp(e)), t.child = null, t.deletions = null, t.sibling = null, t.tag === 5 && (e = t.stateNode, e !== null && (delete e[En], delete e[ia], delete e[xl], delete e[W1], delete e[X1])), t.stateNode = null, t.return = null, t.dependencies = null, t.memoizedProps = null, t.memoizedState = null, t.pendingProps = null, t.stateNode = null, t.updateQueue = null;
}
function Cp(t) {
  return t.tag === 5 || t.tag === 3 || t.tag === 4;
}
function Sh(t) {
  t: for (; ;) {
    for (; t.sibling === null;) {
      if (t.return === null || Cp(t.return)) return null;
      t = t.return;
    }
    for (t.sibling.return = t.return, t = t.sibling; t.tag !== 5 && t.tag !== 6 && t.tag !== 18;) {
      if (t.flags & 2 || t.child === null || t.tag === 4) continue t;
      t.child.return = t, t = t.child;
    }
    if (!(t.flags & 2)) return t.stateNode;
  }
}
function Ll(t, e, n) {
  var o = t.tag;
  if (o === 5 || o === 6) t = t.stateNode, e ? n.nodeType === 8 ? n.parentNode.insertBefore(t, e) : n.insertBefore(t, e) : (n.nodeType === 8 ? (e = n.parentNode, e.insertBefore(t, n)) : (e = n, e.appendChild(t)), n = n._reactRootContainer, n != null || e.onclick !== null || (e.onclick = vo));
  else if (o !== 4 && (t = t.child, t !== null)) for (Ll(t, e, n), t = t.sibling; t !== null;) Ll(t, e, n), t = t.sibling;
}
function Fl(t, e, n) {
  var o = t.tag;
  if (o === 5 || o === 6) t = t.stateNode, e ? n.insertBefore(t, e) : n.appendChild(t);
  else if (o !== 4 && (t = t.child, t !== null)) for (Fl(t, e, n), t = t.sibling; t !== null;) Fl(t, e, n), t = t.sibling;
}
var pe = null, fn = !1;
function Yn(t, e, n) {
  for (n = n.child; n !== null;) Sp(t, e, n), n = n.sibling;
}
function Sp(t, e, n) {
  if (Mn && typeof Mn.onCommitFiberUnmount == "function") try {
    Mn.onCommitFiberUnmount(No, n);
  } catch {
  }
  switch (n.tag) {
    case 5:
      xe || Zi(n, e);
    case 6:
      var o = pe, d = fn;
      pe = null, Yn(t, e, n), pe = o, fn = d, pe !== null && (fn ? (t = pe, n = n.stateNode, t.nodeType === 8 ? t.parentNode.removeChild(n) : t.removeChild(n)) : pe.removeChild(n.stateNode));
      break;
    case 18:
      pe !== null && (fn ? (t = pe, n = n.stateNode, t.nodeType === 8 ? Ts(t.parentNode, n) : t.nodeType === 1 && Ts(t, n), Jr(t)) : Ts(pe, n.stateNode));
      break;
    case 4:
      o = pe, d = fn, pe = n.stateNode.containerInfo, fn = !0, Yn(t, e, n), pe = o, fn = d;
      break;
    case 0:
    case 11:
    case 14:
    case 15:
      if (!xe && (o = n.updateQueue, o !== null && (o = o.lastEffect, o !== null))) {
        d = o = o.next;
        do {
          var g = d, b = g.destroy;
          g = g.tag, b !== void 0 && (g & 2 || g & 4) && _p(n, e, b), d = d.next;
        } while (d !== o);
      }
      Yn(t, e, n);
      break;
    case 1:
      if (!xe && (Zi(n, e), o = n.stateNode, typeof o.componentWillUnmount == "function")) try {
        o.props = n.memoizedProps, o.state = n.memoizedState, o.componentWillUnmount();
      } catch (_) {
        re(n, e, _);
      }
      Yn(t, e, n);
      break;
    case 21:
      Yn(t, e, n);
      break;
    case 22:
      n.mode & 1 ? (xe = (o = xe) || n.memoizedState !== null, Yn(t, e, n), xe = o) : Yn(t, e, n);
      break;
    default:
      Yn(t, e, n);
  }
}
function Eh(t) {
  var e = t.updateQueue;
  if (e !== null) {
    t.updateQueue = null;
    var n = t.stateNode;
    n === null && (n = t.stateNode = new ov()), e.forEach(function (o) {
      var d = mv.bind(null, t, o);
      n.has(o) || (n.add(o), o.then(d, d));
    });
  }
}
function cn(t, e) {
  var n = e.deletions;
  if (n !== null) for (var o = 0; o < n.length; o++) {
    var d = n[o];
    try {
      var g = t, b = e, _ = b;
      t: for (; _ !== null;) {
        switch (_.tag) {
          case 5:
            pe = _.stateNode, fn = !1;
            break t;
          case 3:
            pe = _.stateNode.containerInfo, fn = !0;
            break t;
          case 4:
            pe = _.stateNode.containerInfo, fn = !0;
            break t;
        }
        _ = _.return;
      }
      if (pe === null) throw Error(pt(160));
      Sp(g, b, d), pe = null, fn = !1;
      var w = d.alternate;
      w !== null && (w.return = null), d.return = null;
    } catch (E) {
      re(d, e, E);
    }
  }
  if (e.subtreeFlags & 12854) for (e = e.child; e !== null;) Ep(e, t), e = e.sibling;
}
function Ep(t, e) {
  var n = t.alternate, o = t.flags;
  switch (t.tag) {
    case 0:
    case 11:
    case 14:
    case 15:
      if (cn(e, t), wn(t), o & 4) {
        try {
          Wr(3, t, t.return), qo(3, t);
        } catch (rt) {
          re(t, t.return, rt);
        }
        try {
          Wr(5, t, t.return);
        } catch (rt) {
          re(t, t.return, rt);
        }
      }
      break;
    case 1:
      cn(e, t), wn(t), o & 512 && n !== null && Zi(n, n.return);
      break;
    case 5:
      if (cn(e, t), wn(t), o & 512 && n !== null && Zi(n, n.return), t.flags & 32) {
        var d = t.stateNode;
        try {
          qr(d, "");
        } catch (rt) {
          re(t, t.return, rt);
        }
      }
      if (o & 4 && (d = t.stateNode, d != null)) {
        var g = t.memoizedProps, b = n !== null ? n.memoizedProps : g, _ = t.type, w = t.updateQueue;
        if (t.updateQueue = null, w !== null) try {
          _ === "input" && g.type === "radio" && g.name != null && Vf(d, g), ol(_, b);
          var E = ol(_, g);
          for (b = 0; b < w.length; b += 2) {
            var j = w[b], H = w[b + 1];
            j === "style" ? Kf(d, H) : j === "dangerouslySetInnerHTML" ? Zf(d, H) : j === "children" ? qr(d, H) : $l(d, j, H, E);
          }
          switch (_) {
            case "input":
              el(d, g);
              break;
            case "textarea":
              Gf(d, g);
              break;
            case "select":
              var G = d._wrapperState.wasMultiple;
              d._wrapperState.wasMultiple = !!g.multiple;
              var J = g.value;
              J != null ? Ji(d, !!g.multiple, J, !1) : G !== !!g.multiple && (g.defaultValue != null ? Ji(
                d,
                !!g.multiple,
                g.defaultValue,
                !0
              ) : Ji(d, !!g.multiple, g.multiple ? [] : "", !1));
          }
          d[ia] = g;
        } catch (rt) {
          re(t, t.return, rt);
        }
      }
      break;
    case 6:
      if (cn(e, t), wn(t), o & 4) {
        if (t.stateNode === null) throw Error(pt(162));
        d = t.stateNode, g = t.memoizedProps;
        try {
          d.nodeValue = g;
        } catch (rt) {
          re(t, t.return, rt);
        }
      }
      break;
    case 3:
      if (cn(e, t), wn(t), o & 4 && n !== null && n.memoizedState.isDehydrated) try {
        Jr(e.containerInfo);
      } catch (rt) {
        re(t, t.return, rt);
      }
      break;
    case 4:
      cn(e, t), wn(t);
      break;
    case 13:
      cn(e, t), wn(t), d = t.child, d.flags & 8192 && (g = d.memoizedState !== null, d.stateNode.isHidden = g, !g || d.alternate !== null && d.alternate.memoizedState !== null || (Dc = ae())), o & 4 && Eh(t);
      break;
    case 22:
      if (j = n !== null && n.memoizedState !== null, t.mode & 1 ? (xe = (E = xe) || j, cn(e, t), xe = E) : cn(e, t), wn(t), o & 8192) {
        if (E = t.memoizedState !== null, (t.stateNode.isHidden = E) && !j && t.mode & 1) for (xt = t, j = t.child; j !== null;) {
          for (H = xt = j; xt !== null;) {
            switch (G = xt, J = G.child, G.tag) {
              case 0:
              case 11:
              case 14:
              case 15:
                Wr(4, G, G.return);
                break;
              case 1:
                Zi(G, G.return);
                var nt = G.stateNode;
                if (typeof nt.componentWillUnmount == "function") {
                  o = G, n = G.return;
                  try {
                    e = o, nt.props = e.memoizedProps, nt.state = e.memoizedState, nt.componentWillUnmount();
                  } catch (rt) {
                    re(o, n, rt);
                  }
                }
                break;
              case 5:
                Zi(G, G.return);
                break;
              case 22:
                if (G.memoizedState !== null) {
                  Th(H);
                  continue;
                }
            }
            J !== null ? (J.return = G, xt = J) : Th(H);
          }
          j = j.sibling;
        }
        t: for (j = null, H = t; ;) {
          if (H.tag === 5) {
            if (j === null) {
              j = H;
              try {
                d = H.stateNode, E ? (g = d.style, typeof g.setProperty == "function" ? g.setProperty("display", "none", "important") : g.display = "none") : (_ = H.stateNode, w = H.memoizedProps.style, b = w != null && w.hasOwnProperty("display") ? w.display : null, _.style.display = Qf("display", b));
              } catch (rt) {
                re(t, t.return, rt);
              }
            }
          } else if (H.tag === 6) {
            if (j === null) try {
              H.stateNode.nodeValue = E ? "" : H.memoizedProps;
            } catch (rt) {
              re(t, t.return, rt);
            }
          } else if ((H.tag !== 22 && H.tag !== 23 || H.memoizedState === null || H === t) && H.child !== null) {
            H.child.return = H, H = H.child;
            continue;
          }
          if (H === t) break t;
          for (; H.sibling === null;) {
            if (H.return === null || H.return === t) break t;
            j === H && (j = null), H = H.return;
          }
          j === H && (j = null), H.sibling.return = H.return, H = H.sibling;
        }
      }
      break;
    case 19:
      cn(e, t), wn(t), o & 4 && Eh(t);
      break;
    case 21:
      break;
    default:
      cn(
        e,
        t
      ), wn(t);
  }
}
function wn(t) {
  var e = t.flags;
  if (e & 2) {
    try {
      t: {
        for (var n = t.return; n !== null;) {
          if (Cp(n)) {
            var o = n;
            break t;
          }
          n = n.return;
        }
        throw Error(pt(160));
      }
      switch (o.tag) {
        case 5:
          var d = o.stateNode;
          o.flags & 32 && (qr(d, ""), o.flags &= -33);
          var g = Sh(t);
          Fl(t, g, d);
          break;
        case 3:
        case 4:
          var b = o.stateNode.containerInfo, _ = Sh(t);
          Ll(t, _, b);
          break;
        default:
          throw Error(pt(161));
      }
    } catch (w) {
      re(t, t.return, w);
    }
    t.flags &= -3;
  }
  e & 4096 && (t.flags &= -4097);
}
function lv(t, e, n) {
  xt = t, kp(t);
}
function kp(t, e, n) {
  for (var o = (t.mode & 1) !== 0; xt !== null;) {
    var d = xt, g = d.child;
    if (d.tag === 22 && o) {
      var b = d.memoizedState !== null || za;
      if (!b) {
        var _ = d.alternate, w = _ !== null && _.memoizedState !== null || xe;
        _ = za;
        var E = xe;
        if (za = b, (xe = w) && !E) for (xt = d; xt !== null;) b = xt, w = b.child, b.tag === 22 && b.memoizedState !== null ? Oh(d) : w !== null ? (w.return = b, xt = w) : Oh(d);
        for (; g !== null;) xt = g, kp(g), g = g.sibling;
        xt = d, za = _, xe = E;
      }
      kh(t);
    } else d.subtreeFlags & 8772 && g !== null ? (g.return = d, xt = g) : kh(t);
  }
}
function kh(t) {
  for (; xt !== null;) {
    var e = xt;
    if (e.flags & 8772) {
      var n = e.alternate;
      try {
        if (e.flags & 8772) switch (e.tag) {
          case 0:
          case 11:
          case 15:
            xe || qo(5, e);
            break;
          case 1:
            var o = e.stateNode;
            if (e.flags & 4 && !xe) if (n === null) o.componentDidMount();
            else {
              var d = e.elementType === e.type ? n.memoizedProps : hn(e.type, n.memoizedProps);
              o.componentDidUpdate(d, n.memoizedState, o.__reactInternalSnapshotBeforeUpdate);
            }
            var g = e.updateQueue;
            g !== null && uh(e, g, o);
            break;
          case 3:
            var b = e.updateQueue;
            if (b !== null) {
              if (n = null, e.child !== null) switch (e.child.tag) {
                case 5:
                  n = e.child.stateNode;
                  break;
                case 1:
                  n = e.child.stateNode;
              }
              uh(e, b, n);
            }
            break;
          case 5:
            var _ = e.stateNode;
            if (n === null && e.flags & 4) {
              n = _;
              var w = e.memoizedProps;
              switch (e.type) {
                case "button":
                case "input":
                case "select":
                case "textarea":
                  w.autoFocus && n.focus();
                  break;
                case "img":
                  w.src && (n.src = w.src);
              }
            }
            break;
          case 6:
            break;
          case 4:
            break;
          case 12:
            break;
          case 13:
            if (e.memoizedState === null) {
              var E = e.alternate;
              if (E !== null) {
                var j = E.memoizedState;
                if (j !== null) {
                  var H = j.dehydrated;
                  H !== null && Jr(H);
                }
              }
            }
            break;
          case 19:
          case 17:
          case 21:
          case 22:
          case 23:
          case 25:
            break;
          default:
            throw Error(pt(163));
        }
        xe || e.flags & 512 && jl(e);
      } catch (G) {
        re(e, e.return, G);
      }
    }
    if (e === t) {
      xt = null;
      break;
    }
    if (n = e.sibling, n !== null) {
      n.return = e.return, xt = n;
      break;
    }
    xt = e.return;
  }
}
function Th(t) {
  for (; xt !== null;) {
    var e = xt;
    if (e === t) {
      xt = null;
      break;
    }
    var n = e.sibling;
    if (n !== null) {
      n.return = e.return, xt = n;
      break;
    }
    xt = e.return;
  }
}
function Oh(t) {
  for (; xt !== null;) {
    var e = xt;
    try {
      switch (e.tag) {
        case 0:
        case 11:
        case 15:
          var n = e.return;
          try {
            qo(4, e);
          } catch (w) {
            re(e, n, w);
          }
          break;
        case 1:
          var o = e.stateNode;
          if (typeof o.componentDidMount == "function") {
            var d = e.return;
            try {
              o.componentDidMount();
            } catch (w) {
              re(e, d, w);
            }
          }
          var g = e.return;
          try {
            jl(e);
          } catch (w) {
            re(e, g, w);
          }
          break;
        case 5:
          var b = e.return;
          try {
            jl(e);
          } catch (w) {
            re(e, b, w);
          }
      }
    } catch (w) {
      re(e, e.return, w);
    }
    if (e === t) {
      xt = null;
      break;
    }
    var _ = e.sibling;
    if (_ !== null) {
      _.return = e.return, xt = _;
      break;
    }
    xt = e.return;
  }
}
var cv = Math.ceil, Oo = Xn.ReactCurrentDispatcher, Mc = Xn.ReactCurrentOwner, tn = Xn.ReactCurrentBatchConfig, Ut = 0, fe = null, oe = null, ge = 0, ze = 0, Qi = ui(0), ce = 0, ca = null, Pi = 0, Zo = 0, Ac = 0, Xr = null, De = null, Dc = 0, ur = 1 / 0, Ln = null, Po = !1, Il = null, ri = null, Na = !1, Jn = null, Mo = 0, Yr = 0, Rl = null, ro = -1, ao = 0;
function Ee() {
  return Ut & 6 ? ae() : ro !== -1 ? ro : ro = ae();
}
function ai(t) {
  return t.mode & 1 ? Ut & 2 && ge !== 0 ? ge & -ge : V1.transition !== null ? (ao === 0 && (ao = cd()), ao) : (t = Xt, t !== 0 || (t = window.event, t = t === void 0 ? 16 : md(t.type)), t) : 1;
}
function vn(t, e, n, o) {
  if (50 < Yr) throw Yr = 0, Rl = null, Error(pt(185));
  fa(t, n, o), (!(Ut & 2) || t !== fe) && (t === fe && (!(Ut & 2) && (Zo |= n), ce === 4 && Zn(t, ge)), Ie(t, o), n === 1 && Ut === 0 && !(e.mode & 1) && (ur = ae() + 500, Yo && hi()));
}
function Ie(t, e) {
  var n = t.callbackNode;
  Vm(t, e);
  var o = fo(t, t === fe ? ge : 0);
  if (o === 0) n !== null && Ru(n), t.callbackNode = null, t.callbackPriority = 0;
  else if (e = o & -o, t.callbackPriority !== e) {
    if (n != null && Ru(n), e === 1) t.tag === 0 ? Y1(Ph.bind(null, t)) : Fd(Ph.bind(null, t)), U1(function () {
      !(Ut & 6) && hi();
    }), n = null;
    else {
      switch (ud(o)) {
        case 1:
          n = rc;
          break;
        case 4:
          n = sd;
          break;
        case 16:
          n = ho;
          break;
        case 536870912:
          n = ld;
          break;
        default:
          n = ho;
      }
      n = Lp(n, Tp.bind(null, t));
    }
    t.callbackPriority = e, t.callbackNode = n;
  }
}
function Tp(t, e) {
  if (ro = -1, ao = 0, Ut & 6) throw Error(pt(327));
  var n = t.callbackNode;
  if (ir() && t.callbackNode !== n) return null;
  var o = fo(t, t === fe ? ge : 0);
  if (o === 0) return null;
  if (o & 30 || o & t.expiredLanes || e) e = Ao(t, o);
  else {
    e = o;
    var d = Ut;
    Ut |= 2;
    var g = Pp();
    (fe !== t || ge !== e) && (Ln = null, ur = ae() + 500, Si(t, e));
    do
      try {
        fv();
        break;
      } catch (_) {
        Op(t, _);
      }
    while (!0);
    vc(), Oo.current = g, Ut = d, oe !== null ? e = 0 : (fe = null, ge = 0, e = ce);
  }
  if (e !== 0) {
    if (e === 2 && (d = hl(t), d !== 0 && (o = d, e = Bl(t, d))), e === 1) throw n = ca, Si(t, 0), Zn(t, o), Ie(t, ae()), n;
    if (e === 6) Zn(t, o);
    else {
      if (d = t.current.alternate, !(o & 30) && !uv(d) && (e = Ao(t, o), e === 2 && (g = hl(t), g !== 0 && (o = g, e = Bl(t, g))), e === 1)) throw n = ca, Si(t, 0), Zn(t, o), Ie(t, ae()), n;
      switch (t.finishedWork = d, t.finishedLanes = o, e) {
        case 0:
        case 1:
          throw Error(pt(345));
        case 2:
          yi(t, De, Ln);
          break;
        case 3:
          if (Zn(t, o), (o & 130023424) === o && (e = Dc + 500 - ae(), 10 < e)) {
            if (fo(t, 0) !== 0) break;
            if (d = t.suspendedLanes, (d & o) !== o) {
              Ee(), t.pingedLanes |= t.suspendedLanes & d;
              break;
            }
            t.timeoutHandle = bl(yi.bind(null, t, De, Ln), e);
            break;
          }
          yi(t, De, Ln);
          break;
        case 4:
          if (Zn(t, o), (o & 4194240) === o) break;
          for (e = t.eventTimes, d = -1; 0 < o;) {
            var b = 31 - mn(o);
            g = 1 << b, b = e[b], b > d && (d = b), o &= ~g;
          }
          if (o = d, o = ae() - o, o = (120 > o ? 120 : 480 > o ? 480 : 1080 > o ? 1080 : 1920 > o ? 1920 : 3e3 > o ? 3e3 : 4320 > o ? 4320 : 1960 * cv(o / 1960)) - o, 10 < o) {
            t.timeoutHandle = bl(yi.bind(null, t, De, Ln), o);
            break;
          }
          yi(t, De, Ln);
          break;
        case 5:
          yi(t, De, Ln);
          break;
        default:
          throw Error(pt(329));
      }
    }
  }
  return Ie(t, ae()), t.callbackNode === n ? Tp.bind(null, t) : null;
}
function Bl(t, e) {
  var n = Xr;
  return t.current.memoizedState.isDehydrated && (Si(t, e).flags |= 256), t = Ao(t, e), t !== 2 && (e = De, De = n, e !== null && zl(e)), t;
}
function zl(t) {
  De === null ? De = t : De.push.apply(De, t);
}
function uv(t) {
  for (var e = t; ;) {
    if (e.flags & 16384) {
      var n = e.updateQueue;
      if (n !== null && (n = n.stores, n !== null)) for (var o = 0; o < n.length; o++) {
        var d = n[o], g = d.getSnapshot;
        d = d.value;
        try {
          if (!yn(g(), d)) return !1;
        } catch {
          return !1;
        }
      }
    }
    if (n = e.child, e.subtreeFlags & 16384 && n !== null) n.return = e, e = n;
    else {
      if (e === t) break;
      for (; e.sibling === null;) {
        if (e.return === null || e.return === t) return !0;
        e = e.return;
      }
      e.sibling.return = e.return, e = e.sibling;
    }
  }
  return !0;
}
function Zn(t, e) {
  for (e &= ~Ac, e &= ~Zo, t.suspendedLanes |= e, t.pingedLanes &= ~e, t = t.expirationTimes; 0 < e;) {
    var n = 31 - mn(e), o = 1 << n;
    t[n] = -1, e &= ~o;
  }
}
function Ph(t) {
  if (Ut & 6) throw Error(pt(327));
  ir();
  var e = fo(t, 0);
  if (!(e & 1)) return Ie(t, ae()), null;
  var n = Ao(t, e);
  if (t.tag !== 0 && n === 2) {
    var o = hl(t);
    o !== 0 && (e = o, n = Bl(t, o));
  }
  if (n === 1) throw n = ca, Si(t, 0), Zn(t, e), Ie(t, ae()), n;
  if (n === 6) throw Error(pt(345));
  return t.finishedWork = t.current.alternate, t.finishedLanes = e, yi(t, De, Ln), Ie(t, ae()), null;
}
function jc(t, e) {
  var n = Ut;
  Ut |= 1;
  try {
    return t(e);
  } finally {
    Ut = n, Ut === 0 && (ur = ae() + 500, Yo && hi());
  }
}
function Mi(t) {
  Jn !== null && Jn.tag === 0 && !(Ut & 6) && ir();
  var e = Ut;
  Ut |= 1;
  var n = tn.transition, o = Xt;
  try {
    if (tn.transition = null, Xt = 1, t) return t();
  } finally {
    Xt = o, tn.transition = n, Ut = e, !(Ut & 6) && hi();
  }
}
function Lc() {
  ze = Qi.current, qt(Qi);
}
function Si(t, e) {
  t.finishedWork = null, t.finishedLanes = 0;
  var n = t.timeoutHandle;
  if (n !== -1 && (t.timeoutHandle = -1, N1(n)), oe !== null) for (n = oe.return; n !== null;) {
    var o = n;
    switch (pc(o), o.tag) {
      case 1:
        o = o.type.childContextTypes, o != null && yo();
        break;
      case 3:
        lr(), qt(Le), qt(_e), Cc();
        break;
      case 5:
        wc(o);
        break;
      case 4:
        lr();
        break;
      case 13:
        qt($t);
        break;
      case 19:
        qt($t);
        break;
      case 10:
        yc(o.type._context);
        break;
      case 22:
      case 23:
        Lc();
    }
    n = n.return;
  }
  if (fe = t, oe = t = oi(t.current, null), ge = ze = e, ce = 0, ca = null, Ac = Zo = Pi = 0, De = Xr = null, wi !== null) {
    for (e = 0; e < wi.length; e++) if (n = wi[e], o = n.interleaved, o !== null) {
      n.interleaved = null;
      var d = o.next, g = n.pending;
      if (g !== null) {
        var b = g.next;
        g.next = d, o.next = b;
      }
      n.pending = o;
    }
    wi = null;
  }
  return t;
}
function Op(t, e) {
  do {
    var n = oe;
    try {
      if (vc(), eo.current = To, ko) {
        for (var o = ee.memoizedState; o !== null;) {
          var d = o.queue;
          d !== null && (d.pending = null), o = o.next;
        }
        ko = !1;
      }
      if (Oi = 0, he = le = ee = null, Hr = !1, oa = 0, Mc.current = null, n === null || n.return === null) {
        ce = 1, ca = e, oe = null;
        break;
      }
      t: {
        var g = t, b = n.return, _ = n, w = e;
        if (e = ge, _.flags |= 32768, w !== null && typeof w == "object" && typeof w.then == "function") {
          var E = w, j = _, H = j.tag;
          if (!(j.mode & 1) && (H === 0 || H === 11 || H === 15)) {
            var G = j.alternate;
            G ? (j.updateQueue = G.updateQueue, j.memoizedState = G.memoizedState, j.lanes = G.lanes) : (j.updateQueue = null, j.memoizedState = null);
          }
          var J = mh(b);
          if (J !== null) {
            J.flags &= -257, vh(J, b, _, g, e), J.mode & 1 && gh(g, E, e), e = J, w = E;
            var nt = e.updateQueue;
            if (nt === null) {
              var rt = /* @__PURE__ */ new Set();
              rt.add(w), e.updateQueue = rt;
            } else nt.add(w);
            break t;
          } else {
            if (!(e & 1)) {
              gh(g, E, e), Fc();
              break t;
            }
            w = Error(pt(426));
          }
        } else if (Qt && _.mode & 1) {
          var vt = mh(b);
          if (vt !== null) {
            !(vt.flags & 65536) && (vt.flags |= 256), vh(vt, b, _, g, e), gc(cr(w, _));
            break t;
          }
        }
        g = w = cr(w, _), ce !== 4 && (ce = 2), Xr === null ? Xr = [g] : Xr.push(g), g = b;
        do {
          switch (g.tag) {
            case 3:
              g.flags |= 65536, e &= -e, g.lanes |= e;
              var Y = up(g, w, e);
              ch(g, Y);
              break t;
            case 1:
              _ = w;
              var L = g.type, A = g.stateNode;
              if (!(g.flags & 128) && (typeof L.getDerivedStateFromError == "function" || A !== null && typeof A.componentDidCatch == "function" && (ri === null || !ri.has(A)))) {
                g.flags |= 65536, e &= -e, g.lanes |= e;
                var tt = hp(g, _, e);
                ch(g, tt);
                break t;
              }
          }
          g = g.return;
        } while (g !== null);
      }
      Ap(n);
    } catch (dt) {
      e = dt, oe === n && n !== null && (oe = n = n.return);
      continue;
    }
    break;
  } while (!0);
}
function Pp() {
  var t = Oo.current;
  return Oo.current = To, t === null ? To : t;
}
function Fc() {
  (ce === 0 || ce === 3 || ce === 2) && (ce = 4), fe === null || !(Pi & 268435455) && !(Zo & 268435455) || Zn(fe, ge);
}
function Ao(t, e) {
  var n = Ut;
  Ut |= 2;
  var o = Pp();
  (fe !== t || ge !== e) && (Ln = null, Si(t, e));
  do
    try {
      hv();
      break;
    } catch (d) {
      Op(t, d);
    }
  while (!0);
  if (vc(), Ut = n, Oo.current = o, oe !== null) throw Error(pt(261));
  return fe = null, ge = 0, ce;
}
function hv() {
  for (; oe !== null;) Mp(oe);
}
function fv() {
  for (; oe !== null && !Rm();) Mp(oe);
}
function Mp(t) {
  var e = jp(t.alternate, t, ze);
  t.memoizedProps = t.pendingProps, e === null ? Ap(t) : oe = e, Mc.current = null;
}
function Ap(t) {
  var e = t;
  do {
    var n = e.alternate;
    if (t = e.return, e.flags & 32768) {
      if (n = av(n, e), n !== null) {
        n.flags &= 32767, oe = n;
        return;
      }
      if (t !== null) t.flags |= 32768, t.subtreeFlags = 0, t.deletions = null;
      else {
        ce = 6, oe = null;
        return;
      }
    } else if (n = rv(n, e, ze), n !== null) {
      oe = n;
      return;
    }
    if (e = e.sibling, e !== null) {
      oe = e;
      return;
    }
    oe = e = t;
  } while (e !== null);
  ce === 0 && (ce = 5);
}
function yi(t, e, n) {
  var o = Xt, d = tn.transition;
  try {
    tn.transition = null, Xt = 1, dv(t, e, n, o);
  } finally {
    tn.transition = d, Xt = o;
  }
  return null;
}
function dv(t, e, n, o) {
  do
    ir();
  while (Jn !== null);
  if (Ut & 6) throw Error(pt(327));
  n = t.finishedWork;
  var d = t.finishedLanes;
  if (n === null) return null;
  if (t.finishedWork = null, t.finishedLanes = 0, n === t.current) throw Error(pt(177));
  t.callbackNode = null, t.callbackPriority = 0;
  var g = n.lanes | n.childLanes;
  if (Gm(t, g), t === fe && (oe = fe = null, ge = 0), !(n.subtreeFlags & 2064) && !(n.flags & 2064) || Na || (Na = !0, Lp(ho, function () {
    return ir(), null;
  })), g = (n.flags & 15990) !== 0, n.subtreeFlags & 15990 || g) {
    g = tn.transition, tn.transition = null;
    var b = Xt;
    Xt = 1;
    var _ = Ut;
    Ut |= 4, Mc.current = null, sv(t, n), Ep(n, t), j1(vl), po = !!ml, vl = ml = null, t.current = n, lv(n), Bm(), Ut = _, Xt = b, tn.transition = g;
  } else t.current = n;
  if (Na && (Na = !1, Jn = t, Mo = d), g = t.pendingLanes, g === 0 && (ri = null), Um(n.stateNode), Ie(t, ae()), e !== null) for (o = t.onRecoverableError, n = 0; n < e.length; n++) d = e[n], o(d.value, { componentStack: d.stack, digest: d.digest });
  if (Po) throw Po = !1, t = Il, Il = null, t;
  return Mo & 1 && t.tag !== 0 && ir(), g = t.pendingLanes, g & 1 ? t === Rl ? Yr++ : (Yr = 0, Rl = t) : Yr = 0, hi(), null;
}
function ir() {
  if (Jn !== null) {
    var t = ud(Mo), e = tn.transition, n = Xt;
    try {
      if (tn.transition = null, Xt = 16 > t ? 16 : t, Jn === null) var o = !1;
      else {
        if (t = Jn, Jn = null, Mo = 0, Ut & 6) throw Error(pt(331));
        var d = Ut;
        for (Ut |= 4, xt = t.current; xt !== null;) {
          var g = xt, b = g.child;
          if (xt.flags & 16) {
            var _ = g.deletions;
            if (_ !== null) {
              for (var w = 0; w < _.length; w++) {
                var E = _[w];
                for (xt = E; xt !== null;) {
                  var j = xt;
                  switch (j.tag) {
                    case 0:
                    case 11:
                    case 15:
                      Wr(8, j, g);
                  }
                  var H = j.child;
                  if (H !== null) H.return = j, xt = H;
                  else for (; xt !== null;) {
                    j = xt;
                    var G = j.sibling, J = j.return;
                    if (wp(j), j === E) {
                      xt = null;
                      break;
                    }
                    if (G !== null) {
                      G.return = J, xt = G;
                      break;
                    }
                    xt = J;
                  }
                }
              }
              var nt = g.alternate;
              if (nt !== null) {
                var rt = nt.child;
                if (rt !== null) {
                  nt.child = null;
                  do {
                    var vt = rt.sibling;
                    rt.sibling = null, rt = vt;
                  } while (rt !== null);
                }
              }
              xt = g;
            }
          }
          if (g.subtreeFlags & 2064 && b !== null) b.return = g, xt = b;
          else t: for (; xt !== null;) {
            if (g = xt, g.flags & 2048) switch (g.tag) {
              case 0:
              case 11:
              case 15:
                Wr(9, g, g.return);
            }
            var Y = g.sibling;
            if (Y !== null) {
              Y.return = g.return, xt = Y;
              break t;
            }
            xt = g.return;
          }
        }
        var L = t.current;
        for (xt = L; xt !== null;) {
          b = xt;
          var A = b.child;
          if (b.subtreeFlags & 2064 && A !== null) A.return = b, xt = A;
          else t: for (b = L; xt !== null;) {
            if (_ = xt, _.flags & 2048) try {
              switch (_.tag) {
                case 0:
                case 11:
                case 15:
                  qo(9, _);
              }
            } catch (dt) {
              re(_, _.return, dt);
            }
            if (_ === b) {
              xt = null;
              break t;
            }
            var tt = _.sibling;
            if (tt !== null) {
              tt.return = _.return, xt = tt;
              break t;
            }
            xt = _.return;
          }
        }
        if (Ut = d, hi(), Mn && typeof Mn.onPostCommitFiberRoot == "function") try {
          Mn.onPostCommitFiberRoot(No, t);
        } catch {
        }
        o = !0;
      }
      return o;
    } finally {
      Xt = n, tn.transition = e;
    }
  }
  return !1;
}
function Mh(t, e, n) {
  e = cr(n, e), e = up(t, e, 1), t = ii(t, e, 1), e = Ee(), t !== null && (fa(t, 1, e), Ie(t, e));
}
function re(t, e, n) {
  if (t.tag === 3) Mh(t, t, n);
  else for (; e !== null;) {
    if (e.tag === 3) {
      Mh(e, t, n);
      break;
    } else if (e.tag === 1) {
      var o = e.stateNode;
      if (typeof e.type.getDerivedStateFromError == "function" || typeof o.componentDidCatch == "function" && (ri === null || !ri.has(o))) {
        t = cr(n, t), t = hp(e, t, 1), e = ii(e, t, 1), t = Ee(), e !== null && (fa(e, 1, t), Ie(e, t));
        break;
      }
    }
    e = e.return;
  }
}
function pv(t, e, n) {
  var o = t.pingCache;
  o !== null && o.delete(e), e = Ee(), t.pingedLanes |= t.suspendedLanes & n, fe === t && (ge & n) === n && (ce === 4 || ce === 3 && (ge & 130023424) === ge && 500 > ae() - Dc ? Si(t, 0) : Ac |= n), Ie(t, e);
}
function Dp(t, e) {
  e === 0 && (t.mode & 1 ? (e = Ma, Ma <<= 1, !(Ma & 130023424) && (Ma = 4194304)) : e = 1);
  var n = Ee();
  t = Hn(t, e), t !== null && (fa(t, e, n), Ie(t, n));
}
function gv(t) {
  var e = t.memoizedState, n = 0;
  e !== null && (n = e.retryLane), Dp(t, n);
}
function mv(t, e) {
  var n = 0;
  switch (t.tag) {
    case 13:
      var o = t.stateNode, d = t.memoizedState;
      d !== null && (n = d.retryLane);
      break;
    case 19:
      o = t.stateNode;
      break;
    default:
      throw Error(pt(314));
  }
  o !== null && o.delete(e), Dp(t, n);
}
var jp;
jp = function (t, e, n) {
  if (t !== null) if (t.memoizedProps !== e.pendingProps || Le.current) je = !0;
  else {
    if (!(t.lanes & n) && !(e.flags & 128)) return je = !1, iv(t, e, n);
    je = !!(t.flags & 131072);
  }
  else je = !1, Qt && e.flags & 1048576 && Id(e, _o, e.index);
  switch (e.lanes = 0, e.tag) {
    case 2:
      var o = e.type;
      io(t, e), t = e.pendingProps;
      var d = ar(e, _e.current);
      nr(e, n), d = Ec(null, e, o, t, d, n);
      var g = kc();
      return e.flags |= 1, typeof d == "object" && d !== null && typeof d.render == "function" && d.$$typeof === void 0 ? (e.tag = 1, e.memoizedState = null, e.updateQueue = null, Fe(o) ? (g = !0, bo(e)) : g = !1, e.memoizedState = d.state !== null && d.state !== void 0 ? d.state : null, xc(e), d.updater = Go, e.stateNode = d, d._reactInternals = e, kl(e, o, t, n), e = Pl(null, e, o, !0, g, n)) : (e.tag = 0, Qt && g && dc(e), Se(null, e, d, n), e = e.child), e;
    case 16:
      o = e.elementType;
      t: {
        switch (io(t, e), t = e.pendingProps, d = o._init, o = d(o._payload), e.type = o, d = e.tag = yv(o), t = hn(o, t), d) {
          case 0:
            e = Ol(null, e, o, t, n);
            break t;
          case 1:
            e = xh(null, e, o, t, n);
            break t;
          case 11:
            e = yh(null, e, o, t, n);
            break t;
          case 14:
            e = bh(null, e, o, hn(o.type, t), n);
            break t;
        }
        throw Error(pt(
          306,
          o,
          ""
        ));
      }
      return e;
    case 0:
      return o = e.type, d = e.pendingProps, d = e.elementType === o ? d : hn(o, d), Ol(t, e, o, d, n);
    case 1:
      return o = e.type, d = e.pendingProps, d = e.elementType === o ? d : hn(o, d), xh(t, e, o, d, n);
    case 3:
      t: {
        if (gp(e), t === null) throw Error(pt(387));
        o = e.pendingProps, g = e.memoizedState, d = g.element, Hd(t, e), So(e, o, null, n);
        var b = e.memoizedState;
        if (o = b.element, g.isDehydrated) if (g = { element: o, isDehydrated: !1, cache: b.cache, pendingSuspenseBoundaries: b.pendingSuspenseBoundaries, transitions: b.transitions }, e.updateQueue.baseState = g, e.memoizedState = g, e.flags & 256) {
          d = cr(Error(pt(423)), e), e = _h(t, e, o, n, d);
          break t;
        } else if (o !== d) {
          d = cr(Error(pt(424)), e), e = _h(t, e, o, n, d);
          break t;
        } else for (Ne = ni(e.stateNode.containerInfo.firstChild), Ue = e, Qt = !0, dn = null, n = Nd(e, null, o, n), e.child = n; n;) n.flags = n.flags & -3 | 4096, n = n.sibling;
        else {
          if (or(), o === d) {
            e = Wn(t, e, n);
            break t;
          }
          Se(t, e, o, n);
        }
        e = e.child;
      }
      return e;
    case 5:
      return Wd(e), t === null && Cl(e), o = e.type, d = e.pendingProps, g = t !== null ? t.memoizedProps : null, b = d.children, yl(o, d) ? b = null : g !== null && yl(o, g) && (e.flags |= 32), pp(t, e), Se(t, e, b, n), e.child;
    case 6:
      return t === null && Cl(e), null;
    case 13:
      return mp(t, e, n);
    case 4:
      return _c(e, e.stateNode.containerInfo), o = e.pendingProps, t === null ? e.child = sr(e, null, o, n) : Se(t, e, o, n), e.child;
    case 11:
      return o = e.type, d = e.pendingProps, d = e.elementType === o ? d : hn(o, d), yh(t, e, o, d, n);
    case 7:
      return Se(t, e, e.pendingProps, n), e.child;
    case 8:
      return Se(t, e, e.pendingProps.children, n), e.child;
    case 12:
      return Se(t, e, e.pendingProps.children, n), e.child;
    case 10:
      t: {
        if (o = e.type._context, d = e.pendingProps, g = e.memoizedProps, b = d.value, Vt(wo, o._currentValue), o._currentValue = b, g !== null) if (yn(g.value, b)) {
          if (g.children === d.children && !Le.current) {
            e = Wn(t, e, n);
            break t;
          }
        } else for (g = e.child, g !== null && (g.return = e); g !== null;) {
          var _ = g.dependencies;
          if (_ !== null) {
            b = g.child;
            for (var w = _.firstContext; w !== null;) {
              if (w.context === o) {
                if (g.tag === 1) {
                  w = zn(-1, n & -n), w.tag = 2;
                  var E = g.updateQueue;
                  if (E !== null) {
                    E = E.shared;
                    var j = E.pending;
                    j === null ? w.next = w : (w.next = j.next, j.next = w), E.pending = w;
                  }
                }
                g.lanes |= n, w = g.alternate, w !== null && (w.lanes |= n), Sl(
                  g.return,
                  n,
                  e
                ), _.lanes |= n;
                break;
              }
              w = w.next;
            }
          } else if (g.tag === 10) b = g.type === e.type ? null : g.child;
          else if (g.tag === 18) {
            if (b = g.return, b === null) throw Error(pt(341));
            b.lanes |= n, _ = b.alternate, _ !== null && (_.lanes |= n), Sl(b, n, e), b = g.sibling;
          } else b = g.child;
          if (b !== null) b.return = g;
          else for (b = g; b !== null;) {
            if (b === e) {
              b = null;
              break;
            }
            if (g = b.sibling, g !== null) {
              g.return = b.return, b = g;
              break;
            }
            b = b.return;
          }
          g = b;
        }
        Se(t, e, d.children, n), e = e.child;
      }
      return e;
    case 9:
      return d = e.type, o = e.pendingProps.children, nr(e, n), d = rn(d), o = o(d), e.flags |= 1, Se(t, e, o, n), e.child;
    case 14:
      return o = e.type, d = hn(o, e.pendingProps), d = hn(o.type, d), bh(t, e, o, d, n);
    case 15:
      return fp(t, e, e.type, e.pendingProps, n);
    case 17:
      return o = e.type, d = e.pendingProps, d = e.elementType === o ? d : hn(o, d), io(t, e), e.tag = 1, Fe(o) ? (t = !0, bo(e)) : t = !1, nr(e, n), cp(e, o, d), kl(e, o, d, n), Pl(null, e, o, !0, t, n);
    case 19:
      return vp(t, e, n);
    case 22:
      return dp(t, e, n);
  }
  throw Error(pt(156, e.tag));
};
function Lp(t, e) {
  return od(t, e);
}
function vv(t, e, n, o) {
  this.tag = t, this.key = n, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.ref = null, this.pendingProps = e, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = o, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
}
function $e(t, e, n, o) {
  return new vv(t, e, n, o);
}
function Ic(t) {
  return t = t.prototype, !(!t || !t.isReactComponent);
}
function yv(t) {
  if (typeof t == "function") return Ic(t) ? 1 : 0;
  if (t != null) {
    if (t = t.$$typeof, t === ec) return 11;
    if (t === nc) return 14;
  }
  return 2;
}
function oi(t, e) {
  var n = t.alternate;
  return n === null ? (n = $e(t.tag, e, t.key, t.mode), n.elementType = t.elementType, n.type = t.type, n.stateNode = t.stateNode, n.alternate = t, t.alternate = n) : (n.pendingProps = e, n.type = t.type, n.flags = 0, n.subtreeFlags = 0, n.deletions = null), n.flags = t.flags & 14680064, n.childLanes = t.childLanes, n.lanes = t.lanes, n.child = t.child, n.memoizedProps = t.memoizedProps, n.memoizedState = t.memoizedState, n.updateQueue = t.updateQueue, e = t.dependencies, n.dependencies = e === null ? null : { lanes: e.lanes, firstContext: e.firstContext }, n.sibling = t.sibling, n.index = t.index, n.ref = t.ref, n;
}
function oo(t, e, n, o, d, g) {
  var b = 2;
  if (o = t, typeof t == "function") Ic(t) && (b = 1);
  else if (typeof t == "string") b = 5;
  else t: switch (t) {
    case Ni:
      return Ei(n.children, d, g, e);
    case tc:
      b = 8, d |= 8;
      break;
    case Qs:
      return t = $e(12, n, e, d | 2), t.elementType = Qs, t.lanes = g, t;
    case Ks:
      return t = $e(13, n, e, d), t.elementType = Ks, t.lanes = g, t;
    case Js:
      return t = $e(19, n, e, d), t.elementType = Js, t.lanes = g, t;
    case Wf:
      return Qo(n, d, g, e);
    default:
      if (typeof t == "object" && t !== null) switch (t.$$typeof) {
        case Uf:
          b = 10;
          break t;
        case Hf:
          b = 9;
          break t;
        case ec:
          b = 11;
          break t;
        case nc:
          b = 14;
          break t;
        case Vn:
          b = 16, o = null;
          break t;
      }
      throw Error(pt(130, t == null ? t : typeof t, ""));
  }
  return e = $e(b, n, e, d), e.elementType = t, e.type = o, e.lanes = g, e;
}
function Ei(t, e, n, o) {
  return t = $e(7, t, o, e), t.lanes = n, t;
}
function Qo(t, e, n, o) {
  return t = $e(22, t, o, e), t.elementType = Wf, t.lanes = n, t.stateNode = { isHidden: !1 }, t;
}
function Fs(t, e, n) {
  return t = $e(6, t, null, e), t.lanes = n, t;
}
function Is(t, e, n) {
  return e = $e(4, t.children !== null ? t.children : [], t.key, e), e.lanes = n, e.stateNode = { containerInfo: t.containerInfo, pendingChildren: null, implementation: t.implementation }, e;
}
function bv(t, e, n, o, d) {
  this.tag = e, this.containerInfo = t, this.finishedWork = this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.pendingContext = this.context = null, this.callbackPriority = 0, this.eventTimes = ms(0), this.expirationTimes = ms(-1), this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = ms(0), this.identifierPrefix = o, this.onRecoverableError = d, this.mutableSourceEagerHydrationData = null;
}
function Rc(t, e, n, o, d, g, b, _, w) {
  return t = new bv(t, e, n, _, w), e === 1 ? (e = 1, g === !0 && (e |= 8)) : e = 0, g = $e(3, null, null, e), t.current = g, g.stateNode = t, g.memoizedState = { element: o, isDehydrated: n, cache: null, transitions: null, pendingSuspenseBoundaries: null }, xc(g), t;
}
function xv(t, e, n) {
  var o = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
  return { $$typeof: zi, key: o == null ? null : "" + o, children: t, containerInfo: e, implementation: n };
}
function Fp(t) {
  if (!t) return li;
  t = t._reactInternals;
  t: {
    if (Di(t) !== t || t.tag !== 1) throw Error(pt(170));
    var e = t;
    do {
      switch (e.tag) {
        case 3:
          e = e.stateNode.context;
          break t;
        case 1:
          if (Fe(e.type)) {
            e = e.stateNode.__reactInternalMemoizedMergedChildContext;
            break t;
          }
      }
      e = e.return;
    } while (e !== null);
    throw Error(pt(171));
  }
  if (t.tag === 1) {
    var n = t.type;
    if (Fe(n)) return Ld(t, n, e);
  }
  return e;
}
function Ip(t, e, n, o, d, g, b, _, w) {
  return t = Rc(n, o, !0, t, d, g, b, _, w), t.context = Fp(null), n = t.current, o = Ee(), d = ai(n), g = zn(o, d), g.callback = e ?? null, ii(n, g, d), t.current.lanes = d, fa(t, d, o), Ie(t, o), t;
}
function Ko(t, e, n, o) {
  var d = e.current, g = Ee(), b = ai(d);
  return n = Fp(n), e.context === null ? e.context = n : e.pendingContext = n, e = zn(g, b), e.payload = { element: t }, o = o === void 0 ? null : o, o !== null && (e.callback = o), t = ii(d, e, b), t !== null && (vn(t, d, b, g), to(t, d, b)), b;
}
function Do(t) {
  if (t = t.current, !t.child) return null;
  switch (t.child.tag) {
    case 5:
      return t.child.stateNode;
    default:
      return t.child.stateNode;
  }
}
function Ah(t, e) {
  if (t = t.memoizedState, t !== null && t.dehydrated !== null) {
    var n = t.retryLane;
    t.retryLane = n !== 0 && n < e ? n : e;
  }
}
function Bc(t, e) {
  Ah(t, e), (t = t.alternate) && Ah(t, e);
}
function _v() {
  return null;
}
var Rp = typeof reportError == "function" ? reportError : function (t) {
  console.error(t);
};
function zc(t) {
  this._internalRoot = t;
}
Jo.prototype.render = zc.prototype.render = function (t) {
  var e = this._internalRoot;
  if (e === null) throw Error(pt(409));
  Ko(t, e, null, null);
};
Jo.prototype.unmount = zc.prototype.unmount = function () {
  var t = this._internalRoot;
  if (t !== null) {
    this._internalRoot = null;
    var e = t.containerInfo;
    Mi(function () {
      Ko(null, t, null, null);
    }), e[Un] = null;
  }
};
function Jo(t) {
  this._internalRoot = t;
}
Jo.prototype.unstable_scheduleHydration = function (t) {
  if (t) {
    var e = dd();
    t = { blockedOn: null, target: t, priority: e };
    for (var n = 0; n < qn.length && e !== 0 && e < qn[n].priority; n++);
    qn.splice(n, 0, t), n === 0 && gd(t);
  }
};
function Nc(t) {
  return !(!t || t.nodeType !== 1 && t.nodeType !== 9 && t.nodeType !== 11);
}
function $o(t) {
  return !(!t || t.nodeType !== 1 && t.nodeType !== 9 && t.nodeType !== 11 && (t.nodeType !== 8 || t.nodeValue !== " react-mount-point-unstable "));
}
function Dh() {
}
function wv(t, e, n, o, d) {
  if (d) {
    if (typeof o == "function") {
      var g = o;
      o = function () {
        var E = Do(b);
        g.call(E);
      };
    }
    var b = Ip(e, o, t, 0, null, !1, !1, "", Dh);
    return t._reactRootContainer = b, t[Un] = b.current, ea(t.nodeType === 8 ? t.parentNode : t), Mi(), b;
  }
  for (; d = t.lastChild;) t.removeChild(d);
  if (typeof o == "function") {
    var _ = o;
    o = function () {
      var E = Do(w);
      _.call(E);
    };
  }
  var w = Rc(t, 0, !1, null, null, !1, !1, "", Dh);
  return t._reactRootContainer = w, t[Un] = w.current, ea(t.nodeType === 8 ? t.parentNode : t), Mi(function () {
    Ko(e, w, n, o);
  }), w;
}
function ts(t, e, n, o, d) {
  var g = n._reactRootContainer;
  if (g) {
    var b = g;
    if (typeof d == "function") {
      var _ = d;
      d = function () {
        var w = Do(b);
        _.call(w);
      };
    }
    Ko(e, b, t, d);
  } else b = wv(n, e, t, d, o);
  return Do(b);
}
hd = function (t) {
  switch (t.tag) {
    case 3:
      var e = t.stateNode;
      if (e.current.memoizedState.isDehydrated) {
        var n = Fr(e.pendingLanes);
        n !== 0 && (ac(e, n | 1), Ie(e, ae()), !(Ut & 6) && (ur = ae() + 500, hi()));
      }
      break;
    case 13:
      Mi(function () {
        var o = Hn(t, 1);
        if (o !== null) {
          var d = Ee();
          vn(o, t, 1, d);
        }
      }), Bc(t, 1);
  }
};
oc = function (t) {
  if (t.tag === 13) {
    var e = Hn(t, 134217728);
    if (e !== null) {
      var n = Ee();
      vn(e, t, 134217728, n);
    }
    Bc(t, 134217728);
  }
};
fd = function (t) {
  if (t.tag === 13) {
    var e = ai(t), n = Hn(t, e);
    if (n !== null) {
      var o = Ee();
      vn(n, t, e, o);
    }
    Bc(t, e);
  }
};
dd = function () {
  return Xt;
};
pd = function (t, e) {
  var n = Xt;
  try {
    return Xt = t, e();
  } finally {
    Xt = n;
  }
};
ll = function (t, e, n) {
  switch (e) {
    case "input":
      if (el(t, n), e = n.name, n.type === "radio" && e != null) {
        for (n = t; n.parentNode;) n = n.parentNode;
        for (n = n.querySelectorAll("input[name=" + JSON.stringify("" + e) + '][type="radio"]'), e = 0; e < n.length; e++) {
          var o = n[e];
          if (o !== t && o.form === t.form) {
            var d = Xo(o);
            if (!d) throw Error(pt(90));
            Yf(o), el(o, d);
          }
        }
      }
      break;
    case "textarea":
      Gf(t, n);
      break;
    case "select":
      e = n.value, e != null && Ji(t, !!n.multiple, e, !1);
  }
};
td = jc;
ed = Mi;
var Cv = { usingClientEntryPoint: !1, Events: [pa, Xi, Xo, Jf, $f, jc] }, Dr = { findFiberByHostInstance: _i, bundleType: 0, version: "18.3.1", rendererPackageName: "react-dom" }, Sv = {
  bundleType: Dr.bundleType, version: Dr.version, rendererPackageName: Dr.rendererPackageName, rendererConfig: Dr.rendererConfig, overrideHookState: null, overrideHookStateDeletePath: null, overrideHookStateRenamePath: null, overrideProps: null, overridePropsDeletePath: null, overridePropsRenamePath: null, setErrorHandler: null, setSuspenseHandler: null, scheduleUpdate: null, currentDispatcherRef: Xn.ReactCurrentDispatcher, findHostInstanceByFiber: function (t) {
    return t = rd(t), t === null ? null : t.stateNode;
  }, findFiberByHostInstance: Dr.findFiberByHostInstance || _v, findHostInstancesForRefresh: null, scheduleRefresh: null, scheduleRoot: null, setRefreshHandler: null, getCurrentFiber: null, reconcilerVersion: "18.3.1-next-f1338f8080-20240426"
};
if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u") {
  var Ua = __REACT_DEVTOOLS_GLOBAL_HOOK__;
  if (!Ua.isDisabled && Ua.supportsFiber) try {
    No = Ua.inject(Sv), Mn = Ua;
  } catch {
  }
}
Xe.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = Cv;
Xe.createPortal = function (t, e) {
  var n = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
  if (!Nc(e)) throw Error(pt(200));
  return xv(t, e, null, n);
};
Xe.createRoot = function (t, e) {
  if (!Nc(t)) throw Error(pt(299));
  var n = !1, o = "", d = Rp;
  return e != null && (e.unstable_strictMode === !0 && (n = !0), e.identifierPrefix !== void 0 && (o = e.identifierPrefix), e.onRecoverableError !== void 0 && (d = e.onRecoverableError)), e = Rc(t, 1, !1, null, null, n, !1, o, d), t[Un] = e.current, ea(t.nodeType === 8 ? t.parentNode : t), new zc(e);
};
Xe.findDOMNode = function (t) {
  if (t == null) return null;
  if (t.nodeType === 1) return t;
  var e = t._reactInternals;
  if (e === void 0)
    throw typeof t.render == "function" ? Error(pt(188)) : (t = Object.keys(t).join(","), Error(pt(268, t)));
  return t = rd(e), t = t === null ? null : t.stateNode, t;
};
Xe.flushSync = function (t) {
  return Mi(t);
};
Xe.hydrate = function (t, e, n) {
  if (!$o(e)) throw Error(pt(200));
  return ts(null, t, e, !0, n);
};
Xe.hydrateRoot = function (t, e, n) {
  if (!Nc(t)) throw Error(pt(405));
  var o = n != null && n.hydratedSources || null, d = !1, g = "", b = Rp;
  if (n != null && (n.unstable_strictMode === !0 && (d = !0), n.identifierPrefix !== void 0 && (g = n.identifierPrefix), n.onRecoverableError !== void 0 && (b = n.onRecoverableError)), e = Ip(e, null, t, 1, n ?? null, d, !1, g, b), t[Un] = e.current, ea(t), o) for (t = 0; t < o.length; t++) n = o[t], d = n._getVersion, d = d(n._source), e.mutableSourceEagerHydrationData == null ? e.mutableSourceEagerHydrationData = [n, d] : e.mutableSourceEagerHydrationData.push(
    n,
    d
  );
  return new Jo(e);
};
Xe.render = function (t, e, n) {
  if (!$o(e)) throw Error(pt(200));
  return ts(null, t, e, !1, n);
};
Xe.unmountComponentAtNode = function (t) {
  if (!$o(t)) throw Error(pt(40));
  return t._reactRootContainer ? (Mi(function () {
    ts(null, null, t, !1, function () {
      t._reactRootContainer = null, t[Un] = null;
    });
  }), !0) : !1;
};
Xe.unstable_batchedUpdates = jc;
Xe.unstable_renderSubtreeIntoContainer = function (t, e, n, o) {
  if (!$o(n)) throw Error(pt(200));
  if (t == null || t._reactInternals === void 0) throw Error(pt(38));
  return ts(t, e, n, !1, o);
};
Xe.version = "18.3.1-next-f1338f8080-20240426";
function Bp() {
  if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function"))
    try {
      __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(Bp);
    } catch (t) {
      console.error(t);
    }
}
Bp(), Rf.exports = Xe;
var Ev = Rf.exports;
const kv = /* @__PURE__ */ Vl(Ev);
function Tv(t, e) {
  return e.forEach(function (n) {
    n && typeof n != "string" && !Array.isArray(n) && Object.keys(n).forEach(function (o) {
      if (o !== "default" && !(o in t)) {
        var d = Object.getOwnPropertyDescriptor(n, o);
        Object.defineProperty(t, o, d.get ? d : {
          enumerable: !0, get: function () {
            return n[o];
          }
        });
      }
    });
  }), Object.freeze(t);
}
var jh = typeof global < "u" ? global : typeof self < "u" ? self : typeof window < "u" ? window : {}, kn = [], qe = [], Ov = typeof Uint8Array < "u" ? Uint8Array : Array, Uc = !1;
function zp() {
  Uc = !0;
  for (var t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", e = 0; e < 64; ++e) kn[e] = t[e], qe[t.charCodeAt(e)] = e;
  qe[45] = 62, qe[95] = 63;
}
function Pv(t, e, n) {
  for (var o, d, g = [], b = e; b < n; b += 3) o = (t[b] << 16) + (t[b + 1] << 8) + t[b + 2], g.push(kn[(d = o) >> 18 & 63] + kn[d >> 12 & 63] + kn[d >> 6 & 63] + kn[63 & d]);
  return g.join("");
}
function Lh(t) {
  var e;
  Uc || zp();
  for (var n = t.length, o = n % 3, d = "", g = [], b = 16383, _ = 0, w = n - o; _ < w; _ += b) g.push(Pv(t, _, _ + b > w ? w : _ + b));
  return o === 1 ? (e = t[n - 1], d += kn[e >> 2], d += kn[e << 4 & 63], d += "==") : o === 2 && (e = (t[n - 2] << 8) + t[n - 1], d += kn[e >> 10], d += kn[e >> 4 & 63], d += kn[e << 2 & 63], d += "="), g.push(d), g.join("");
}
function Ha(t, e, n, o, d) {
  var g, b, _ = 8 * d - o - 1, w = (1 << _) - 1, E = w >> 1, j = -7, H = n ? d - 1 : 0, G = n ? -1 : 1, J = t[e + H];
  for (H += G, g = J & (1 << -j) - 1, J >>= -j, j += _; j > 0; g = 256 * g + t[e + H], H += G, j -= 8);
  for (b = g & (1 << -j) - 1, g >>= -j, j += o; j > 0; b = 256 * b + t[e + H], H += G, j -= 8);
  if (g === 0) g = 1 - E;
  else {
    if (g === w) return b ? NaN : 1 / 0 * (J ? -1 : 1);
    b += Math.pow(2, o), g -= E;
  }
  return (J ? -1 : 1) * b * Math.pow(2, g - o);
}
function Np(t, e, n, o, d, g) {
  var b, _, w, E = 8 * g - d - 1, j = (1 << E) - 1, H = j >> 1, G = d === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0, J = o ? 0 : g - 1, nt = o ? 1 : -1, rt = e < 0 || e === 0 && 1 / e < 0 ? 1 : 0;
  for (e = Math.abs(e), isNaN(e) || e === 1 / 0 ? (_ = isNaN(e) ? 1 : 0, b = j) : (b = Math.floor(Math.log(e) / Math.LN2), e * (w = Math.pow(2, -b)) < 1 && (b--, w *= 2), (e += b + H >= 1 ? G / w : G * Math.pow(2, 1 - H)) * w >= 2 && (b++, w /= 2), b + H >= j ? (_ = 0, b = j) : b + H >= 1 ? (_ = (e * w - 1) * Math.pow(2, d), b += H) : (_ = e * Math.pow(2, H - 1) * Math.pow(2, d), b = 0)); d >= 8; t[n + J] = 255 & _, J += nt, _ /= 256, d -= 8);
  for (b = b << d | _, E += d; E > 0; t[n + J] = 255 & b, J += nt, b /= 256, E -= 8);
  t[n + J - nt] |= 128 * rt;
}
var Mv = {}.toString, Up = Array.isArray || function (t) {
  return Mv.call(t) == "[object Array]";
};
function jo() {
  return ut.TYPED_ARRAY_SUPPORT ? 2147483647 : 1073741823;
}
function Bn(t, e) {
  if (jo() < e) throw new RangeError("Invalid typed array length");
  return ut.TYPED_ARRAY_SUPPORT ? (t = new Uint8Array(e)).__proto__ = ut.prototype : (t === null && (t = new ut(e)), t.length = e), t;
}
function ut(t, e, n) {
  if (!(ut.TYPED_ARRAY_SUPPORT || this instanceof ut)) return new ut(t, e, n);
  if (typeof t == "number") {
    if (typeof e == "string") throw new Error("If encoding is specified then the first argument must be a string");
    return Nl(this, t);
  }
  return Hp(this, t, e, n);
}
function Hp(t, e, n, o) {
  if (typeof e == "number") throw new TypeError('"value" argument must not be a number');
  return typeof ArrayBuffer < "u" && e instanceof ArrayBuffer ? function (d, g, b, _) {
    if (g.byteLength, b < 0 || g.byteLength < b) throw new RangeError("'offset' is out of bounds");
    if (g.byteLength < b + (_ || 0)) throw new RangeError("'length' is out of bounds");
    return g = b === void 0 && _ === void 0 ? new Uint8Array(g) : _ === void 0 ? new Uint8Array(g, b) : new Uint8Array(g, b, _), ut.TYPED_ARRAY_SUPPORT ? (d = g).__proto__ = ut.prototype : d = Rs(d, g), d;
  }(t, e, n, o) : typeof e == "string" ? function (d, g, b) {
    if (typeof b == "string" && b !== "" || (b = "utf8"), !ut.isEncoding(b)) throw new TypeError('"encoding" must be a valid string encoding');
    var _ = 0 | Xp(g, b);
    d = Bn(d, _);
    var w = d.write(g, b);
    return w !== _ && (d = d.slice(0, w)), d;
  }(t, e, n) : function (d, g) {
    if (Tn(g)) {
      var b = 0 | Hc(g.length);
      return (d = Bn(d, b)).length === 0 || g.copy(d, 0, 0, b), d;
    }
    if (g) {
      if (typeof ArrayBuffer < "u" && g.buffer instanceof ArrayBuffer || "length" in g) return typeof g.length != "number" || (_ = g.length) != _ ? Bn(d, 0) : Rs(d, g);
      if (g.type === "Buffer" && Up(g.data)) return Rs(d, g.data);
    }
    var _;
    throw new TypeError("First argument must be a string, Buffer, ArrayBuffer, Array, or array-like object.");
  }(t, e);
}
function Wp(t) {
  if (typeof t != "number") throw new TypeError('"size" argument must be a number');
  if (t < 0) throw new RangeError('"size" argument must not be negative');
}
function Nl(t, e) {
  if (Wp(e), t = Bn(t, e < 0 ? 0 : 0 | Hc(e)), !ut.TYPED_ARRAY_SUPPORT) for (var n = 0; n < e; ++n) t[n] = 0;
  return t;
}
function Rs(t, e) {
  var n = e.length < 0 ? 0 : 0 | Hc(e.length);
  t = Bn(t, n);
  for (var o = 0; o < n; o += 1) t[o] = 255 & e[o];
  return t;
}
function Hc(t) {
  if (t >= jo()) throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x" + jo().toString(16) + " bytes");
  return 0 | t;
}
function Tn(t) {
  return !(t == null || !t._isBuffer);
}
function Xp(t, e) {
  if (Tn(t)) return t.length;
  if (typeof ArrayBuffer < "u" && typeof ArrayBuffer.isView == "function" && (ArrayBuffer.isView(t) || t instanceof ArrayBuffer)) return t.byteLength;
  typeof t != "string" && (t = "" + t);
  var n = t.length;
  if (n === 0) return 0;
  for (var o = !1; ;) switch (e) {
    case "ascii":
    case "latin1":
    case "binary":
      return n;
    case "utf8":
    case "utf-8":
    case void 0:
      return Lo(t).length;
    case "ucs2":
    case "ucs-2":
    case "utf16le":
    case "utf-16le":
      return 2 * n;
    case "hex":
      return n >>> 1;
    case "base64":
      return qp(t).length;
    default:
      if (o) return Lo(t).length;
      e = ("" + e).toLowerCase(), o = !0;
  }
}
function Av(t, e, n) {
  var o = !1;
  if ((e === void 0 || e < 0) && (e = 0), e > this.length || ((n === void 0 || n > this.length) && (n = this.length), n <= 0) || (n >>>= 0) <= (e >>>= 0)) return "";
  for (t || (t = "utf8"); ;) switch (t) {
    case "hex":
      return Nv(this, e, n);
    case "utf8":
    case "utf-8":
      return Vp(this, e, n);
    case "ascii":
      return Bv(this, e, n);
    case "latin1":
    case "binary":
      return zv(this, e, n);
    case "base64":
      return Rv(this, e, n);
    case "ucs2":
    case "ucs-2":
    case "utf16le":
    case "utf-16le":
      return Uv(this, e, n);
    default:
      if (o) throw new TypeError("Unknown encoding: " + t);
      t = (t + "").toLowerCase(), o = !0;
  }
}
function gi(t, e, n) {
  var o = t[e];
  t[e] = t[n], t[n] = o;
}
function Fh(t, e, n, o, d) {
  if (t.length === 0) return -1;
  if (typeof n == "string" ? (o = n, n = 0) : n > 2147483647 ? n = 2147483647 : n < -2147483648 && (n = -2147483648), n = +n, isNaN(n) && (n = d ? 0 : t.length - 1), n < 0 && (n = t.length + n), n >= t.length) {
    if (d) return -1;
    n = t.length - 1;
  } else if (n < 0) {
    if (!d) return -1;
    n = 0;
  }
  if (typeof e == "string" && (e = ut.from(e, o)), Tn(e)) return e.length === 0 ? -1 : Ih(t, e, n, o, d);
  if (typeof e == "number") return e &= 255, ut.TYPED_ARRAY_SUPPORT && typeof Uint8Array.prototype.indexOf == "function" ? d ? Uint8Array.prototype.indexOf.call(t, e, n) : Uint8Array.prototype.lastIndexOf.call(t, e, n) : Ih(t, [e], n, o, d);
  throw new TypeError("val must be string, number or Buffer");
}
function Ih(t, e, n, o, d) {
  var g, b = 1, _ = t.length, w = e.length;
  if (o !== void 0 && ((o = String(o).toLowerCase()) === "ucs2" || o === "ucs-2" || o === "utf16le" || o === "utf-16le")) {
    if (t.length < 2 || e.length < 2) return -1;
    b = 2, _ /= 2, w /= 2, n /= 2;
  }
  function E(J, nt) {
    return b === 1 ? J[nt] : J.readUInt16BE(nt * b);
  }
  if (d) {
    var j = -1;
    for (g = n; g < _; g++) if (E(t, g) === E(e, j === -1 ? 0 : g - j)) {
      if (j === -1 && (j = g), g - j + 1 === w) return j * b;
    } else j !== -1 && (g -= g - j), j = -1;
  } else for (n + w > _ && (n = _ - w), g = n; g >= 0; g--) {
    for (var H = !0, G = 0; G < w; G++) if (E(t, g + G) !== E(e, G)) {
      H = !1;
      break;
    }
    if (H) return g;
  }
  return -1;
}
function Dv(t, e, n, o) {
  n = Number(n) || 0;
  var d = t.length - n;
  o ? (o = Number(o)) > d && (o = d) : o = d;
  var g = e.length;
  if (g % 2 != 0) throw new TypeError("Invalid hex string");
  o > g / 2 && (o = g / 2);
  for (var b = 0; b < o; ++b) {
    var _ = parseInt(e.substr(2 * b, 2), 16);
    if (isNaN(_)) return b;
    t[n + b] = _;
  }
  return b;
}
function jv(t, e, n, o) {
  return es(Lo(e, t.length - n), t, n, o);
}
function Yp(t, e, n, o) {
  return es(function (d) {
    for (var g = [], b = 0; b < d.length; ++b) g.push(255 & d.charCodeAt(b));
    return g;
  }(e), t, n, o);
}
function Lv(t, e, n, o) {
  return Yp(t, e, n, o);
}
function Fv(t, e, n, o) {
  return es(qp(e), t, n, o);
}
function Iv(t, e, n, o) {
  return es(function (d, g) {
    for (var b, _, w, E = [], j = 0; j < d.length && !((g -= 2) < 0); ++j) _ = (b = d.charCodeAt(j)) >> 8, w = b % 256, E.push(w), E.push(_);
    return E;
  }(e, t.length - n), t, n, o);
}
function Rv(t, e, n) {
  return e === 0 && n === t.length ? Lh(t) : Lh(t.slice(e, n));
}
function Vp(t, e, n) {
  n = Math.min(t.length, n);
  for (var o = [], d = e; d < n;) {
    var g, b, _, w, E = t[d], j = null, H = E > 239 ? 4 : E > 223 ? 3 : E > 191 ? 2 : 1;
    if (d + H <= n) switch (H) {
      case 1:
        E < 128 && (j = E);
        break;
      case 2:
        (192 & (g = t[d + 1])) == 128 && (w = (31 & E) << 6 | 63 & g) > 127 && (j = w);
        break;
      case 3:
        g = t[d + 1], b = t[d + 2], (192 & g) == 128 && (192 & b) == 128 && (w = (15 & E) << 12 | (63 & g) << 6 | 63 & b) > 2047 && (w < 55296 || w > 57343) && (j = w);
        break;
      case 4:
        g = t[d + 1], b = t[d + 2], _ = t[d + 3], (192 & g) == 128 && (192 & b) == 128 && (192 & _) == 128 && (w = (15 & E) << 18 | (63 & g) << 12 | (63 & b) << 6 | 63 & _) > 65535 && w < 1114112 && (j = w);
    }
    j === null ? (j = 65533, H = 1) : j > 65535 && (j -= 65536, o.push(j >>> 10 & 1023 | 55296), j = 56320 | 1023 & j), o.push(j), d += H;
  }
  return function (G) {
    var J = G.length;
    if (J <= Rh) return String.fromCharCode.apply(String, G);
    for (var nt = "", rt = 0; rt < J;) nt += String.fromCharCode.apply(String, G.slice(rt, rt += Rh));
    return nt;
  }(o);
}
ut.TYPED_ARRAY_SUPPORT = jh.TYPED_ARRAY_SUPPORT === void 0 || jh.TYPED_ARRAY_SUPPORT, jo(), ut.poolSize = 8192, ut._augment = function (t) {
  return t.__proto__ = ut.prototype, t;
}, ut.from = function (t, e, n) {
  return Hp(null, t, e, n);
}, ut.TYPED_ARRAY_SUPPORT && (ut.prototype.__proto__ = Uint8Array.prototype, ut.__proto__ = Uint8Array, typeof Symbol < "u" && Symbol.species && ut[Symbol.species]), ut.alloc = function (t, e, n) {
  return function (o, d, g, b) {
    return Wp(d), d <= 0 ? Bn(o, d) : g !== void 0 ? typeof b == "string" ? Bn(o, d).fill(g, b) : Bn(o, d).fill(g) : Bn(o, d);
  }(null, t, e, n);
}, ut.allocUnsafe = function (t) {
  return Nl(null, t);
}, ut.allocUnsafeSlow = function (t) {
  return Nl(null, t);
}, ut.isBuffer = function (t) {
  return t != null && (!!t._isBuffer || Nh(t) || function (e) {
    return typeof e.readFloatLE == "function" && typeof e.slice == "function" && Nh(e.slice(0, 0));
  }(t));
}, ut.compare = function (t, e) {
  if (!Tn(t) || !Tn(e)) throw new TypeError("Arguments must be Buffers");
  if (t === e) return 0;
  for (var n = t.length, o = e.length, d = 0, g = Math.min(n, o); d < g; ++d) if (t[d] !== e[d]) {
    n = t[d], o = e[d];
    break;
  }
  return n < o ? -1 : o < n ? 1 : 0;
}, ut.isEncoding = function (t) {
  switch (String(t).toLowerCase()) {
    case "hex":
    case "utf8":
    case "utf-8":
    case "ascii":
    case "latin1":
    case "binary":
    case "base64":
    case "ucs2":
    case "ucs-2":
    case "utf16le":
    case "utf-16le":
      return !0;
    default:
      return !1;
  }
}, ut.concat = function (t, e) {
  if (!Up(t)) throw new TypeError('"list" argument must be an Array of Buffers');
  if (t.length === 0) return ut.alloc(0);
  var n;
  if (e === void 0) for (e = 0, n = 0; n < t.length; ++n) e += t[n].length;
  var o = ut.allocUnsafe(e), d = 0;
  for (n = 0; n < t.length; ++n) {
    var g = t[n];
    if (!Tn(g)) throw new TypeError('"list" argument must be an Array of Buffers');
    g.copy(o, d), d += g.length;
  }
  return o;
}, ut.byteLength = Xp, ut.prototype._isBuffer = !0, ut.prototype.swap16 = function () {
  var t = this.length;
  if (t % 2 != 0) throw new RangeError("Buffer size must be a multiple of 16-bits");
  for (var e = 0; e < t; e += 2) gi(this, e, e + 1);
  return this;
}, ut.prototype.swap32 = function () {
  var t = this.length;
  if (t % 4 != 0) throw new RangeError("Buffer size must be a multiple of 32-bits");
  for (var e = 0; e < t; e += 4) gi(this, e, e + 3), gi(this, e + 1, e + 2);
  return this;
}, ut.prototype.swap64 = function () {
  var t = this.length;
  if (t % 8 != 0) throw new RangeError("Buffer size must be a multiple of 64-bits");
  for (var e = 0; e < t; e += 8) gi(this, e, e + 7), gi(this, e + 1, e + 6), gi(this, e + 2, e + 5), gi(this, e + 3, e + 4);
  return this;
}, ut.prototype.toString = function () {
  var t = 0 | this.length;
  return t === 0 ? "" : arguments.length === 0 ? Vp(this, 0, t) : Av.apply(this, arguments);
}, ut.prototype.equals = function (t) {
  if (!Tn(t)) throw new TypeError("Argument must be a Buffer");
  return this === t || ut.compare(this, t) === 0;
}, ut.prototype.inspect = function () {
  var t = "";
  return this.length > 0 && (t = this.toString("hex", 0, 50).match(/.{2}/g).join(" "), this.length > 50 && (t += " ... ")), "<Buffer " + t + ">";
}, ut.prototype.compare = function (t, e, n, o, d) {
  if (!Tn(t)) throw new TypeError("Argument must be a Buffer");
  if (e === void 0 && (e = 0), n === void 0 && (n = t ? t.length : 0), o === void 0 && (o = 0), d === void 0 && (d = this.length), e < 0 || n > t.length || o < 0 || d > this.length) throw new RangeError("out of range index");
  if (o >= d && e >= n) return 0;
  if (o >= d) return -1;
  if (e >= n) return 1;
  if (this === t) return 0;
  for (var g = (d >>>= 0) - (o >>>= 0), b = (n >>>= 0) - (e >>>= 0), _ = Math.min(g, b), w = this.slice(o, d), E = t.slice(e, n), j = 0; j < _; ++j) if (w[j] !== E[j]) {
    g = w[j], b = E[j];
    break;
  }
  return g < b ? -1 : b < g ? 1 : 0;
}, ut.prototype.includes = function (t, e, n) {
  return this.indexOf(t, e, n) !== -1;
}, ut.prototype.indexOf = function (t, e, n) {
  return Fh(this, t, e, n, !0);
}, ut.prototype.lastIndexOf = function (t, e, n) {
  return Fh(this, t, e, n, !1);
}, ut.prototype.write = function (t, e, n, o) {
  if (e === void 0) o = "utf8", n = this.length, e = 0;
  else if (n === void 0 && typeof e == "string") o = e, n = this.length, e = 0;
  else {
    if (!isFinite(e)) throw new Error("Buffer.write(string, encoding, offset[, length]) is no longer supported");
    e |= 0, isFinite(n) ? (n |= 0, o === void 0 && (o = "utf8")) : (o = n, n = void 0);
  }
  var d = this.length - e;
  if ((n === void 0 || n > d) && (n = d), t.length > 0 && (n < 0 || e < 0) || e > this.length) throw new RangeError("Attempt to write outside buffer bounds");
  o || (o = "utf8");
  for (var g = !1; ;) switch (o) {
    case "hex":
      return Dv(this, t, e, n);
    case "utf8":
    case "utf-8":
      return jv(this, t, e, n);
    case "ascii":
      return Yp(this, t, e, n);
    case "latin1":
    case "binary":
      return Lv(this, t, e, n);
    case "base64":
      return Fv(this, t, e, n);
    case "ucs2":
    case "ucs-2":
    case "utf16le":
    case "utf-16le":
      return Iv(this, t, e, n);
    default:
      if (g) throw new TypeError("Unknown encoding: " + o);
      o = ("" + o).toLowerCase(), g = !0;
  }
}, ut.prototype.toJSON = function () {
  return { type: "Buffer", data: Array.prototype.slice.call(this._arr || this, 0) };
};
var Rh = 4096;
function Bv(t, e, n) {
  var o = "";
  n = Math.min(t.length, n);
  for (var d = e; d < n; ++d) o += String.fromCharCode(127 & t[d]);
  return o;
}
function zv(t, e, n) {
  var o = "";
  n = Math.min(t.length, n);
  for (var d = e; d < n; ++d) o += String.fromCharCode(t[d]);
  return o;
}
function Nv(t, e, n) {
  var o = t.length;
  (!e || e < 0) && (e = 0), (!n || n < 0 || n > o) && (n = o);
  for (var d = "", g = e; g < n; ++g) d += Wv(t[g]);
  return d;
}
function Uv(t, e, n) {
  for (var o = t.slice(e, n), d = "", g = 0; g < o.length; g += 2) d += String.fromCharCode(o[g] + 256 * o[g + 1]);
  return d;
}
function ue(t, e, n) {
  if (t % 1 != 0 || t < 0) throw new RangeError("offset is not uint");
  if (t + e > n) throw new RangeError("Trying to access beyond buffer length");
}
function Ae(t, e, n, o, d, g) {
  if (!Tn(t)) throw new TypeError('"buffer" argument must be a Buffer instance');
  if (e > d || e < g) throw new RangeError('"value" argument is out of bounds');
  if (n + o > t.length) throw new RangeError("Index out of range");
}
function Wa(t, e, n, o) {
  e < 0 && (e = 65535 + e + 1);
  for (var d = 0, g = Math.min(t.length - n, 2); d < g; ++d) t[n + d] = (e & 255 << 8 * (o ? d : 1 - d)) >>> 8 * (o ? d : 1 - d);
}
function Xa(t, e, n, o) {
  e < 0 && (e = 4294967295 + e + 1);
  for (var d = 0, g = Math.min(t.length - n, 4); d < g; ++d) t[n + d] = e >>> 8 * (o ? d : 3 - d) & 255;
}
function Gp(t, e, n, o, d, g) {
  if (n + o > t.length) throw new RangeError("Index out of range");
  if (n < 0) throw new RangeError("Index out of range");
}
function Bh(t, e, n, o, d) {
  return d || Gp(t, 0, n, 4), Np(t, e, n, o, 23, 4), n + 4;
}
function zh(t, e, n, o, d) {
  return d || Gp(t, 0, n, 8), Np(t, e, n, o, 52, 8), n + 8;
}
ut.prototype.slice = function (t, e) {
  var n, o = this.length;
  if ((t = ~~t) < 0 ? (t += o) < 0 && (t = 0) : t > o && (t = o), (e = e === void 0 ? o : ~~e) < 0 ? (e += o) < 0 && (e = 0) : e > o && (e = o), e < t && (e = t), ut.TYPED_ARRAY_SUPPORT) (n = this.subarray(t, e)).__proto__ = ut.prototype;
  else {
    var d = e - t;
    n = new ut(d, void 0);
    for (var g = 0; g < d; ++g) n[g] = this[g + t];
  }
  return n;
}, ut.prototype.readUIntLE = function (t, e, n) {
  t |= 0, e |= 0, n || ue(t, e, this.length);
  for (var o = this[t], d = 1, g = 0; ++g < e && (d *= 256);) o += this[t + g] * d;
  return o;
}, ut.prototype.readUIntBE = function (t, e, n) {
  t |= 0, e |= 0, n || ue(t, e, this.length);
  for (var o = this[t + --e], d = 1; e > 0 && (d *= 256);) o += this[t + --e] * d;
  return o;
}, ut.prototype.readUInt8 = function (t, e) {
  return e || ue(t, 1, this.length), this[t];
}, ut.prototype.readUInt16LE = function (t, e) {
  return e || ue(t, 2, this.length), this[t] | this[t + 1] << 8;
}, ut.prototype.readUInt16BE = function (t, e) {
  return e || ue(t, 2, this.length), this[t] << 8 | this[t + 1];
}, ut.prototype.readUInt32LE = function (t, e) {
  return e || ue(t, 4, this.length), (this[t] | this[t + 1] << 8 | this[t + 2] << 16) + 16777216 * this[t + 3];
}, ut.prototype.readUInt32BE = function (t, e) {
  return e || ue(t, 4, this.length), 16777216 * this[t] + (this[t + 1] << 16 | this[t + 2] << 8 | this[t + 3]);
}, ut.prototype.readIntLE = function (t, e, n) {
  t |= 0, e |= 0, n || ue(t, e, this.length);
  for (var o = this[t], d = 1, g = 0; ++g < e && (d *= 256);) o += this[t + g] * d;
  return o >= (d *= 128) && (o -= Math.pow(2, 8 * e)), o;
}, ut.prototype.readIntBE = function (t, e, n) {
  t |= 0, e |= 0, n || ue(t, e, this.length);
  for (var o = e, d = 1, g = this[t + --o]; o > 0 && (d *= 256);) g += this[t + --o] * d;
  return g >= (d *= 128) && (g -= Math.pow(2, 8 * e)), g;
}, ut.prototype.readInt8 = function (t, e) {
  return e || ue(t, 1, this.length), 128 & this[t] ? -1 * (255 - this[t] + 1) : this[t];
}, ut.prototype.readInt16LE = function (t, e) {
  e || ue(t, 2, this.length);
  var n = this[t] | this[t + 1] << 8;
  return 32768 & n ? 4294901760 | n : n;
}, ut.prototype.readInt16BE = function (t, e) {
  e || ue(t, 2, this.length);
  var n = this[t + 1] | this[t] << 8;
  return 32768 & n ? 4294901760 | n : n;
}, ut.prototype.readInt32LE = function (t, e) {
  return e || ue(t, 4, this.length), this[t] | this[t + 1] << 8 | this[t + 2] << 16 | this[t + 3] << 24;
}, ut.prototype.readInt32BE = function (t, e) {
  return e || ue(t, 4, this.length), this[t] << 24 | this[t + 1] << 16 | this[t + 2] << 8 | this[t + 3];
}, ut.prototype.readFloatLE = function (t, e) {
  return e || ue(t, 4, this.length), Ha(this, t, !0, 23, 4);
}, ut.prototype.readFloatBE = function (t, e) {
  return e || ue(t, 4, this.length), Ha(this, t, !1, 23, 4);
}, ut.prototype.readDoubleLE = function (t, e) {
  return e || ue(t, 8, this.length), Ha(this, t, !0, 52, 8);
}, ut.prototype.readDoubleBE = function (t, e) {
  return e || ue(t, 8, this.length), Ha(this, t, !1, 52, 8);
}, ut.prototype.writeUIntLE = function (t, e, n, o) {
  t = +t, e |= 0, n |= 0, o || Ae(this, t, e, n, Math.pow(2, 8 * n) - 1, 0);
  var d = 1, g = 0;
  for (this[e] = 255 & t; ++g < n && (d *= 256);) this[e + g] = t / d & 255;
  return e + n;
}, ut.prototype.writeUIntBE = function (t, e, n, o) {
  t = +t, e |= 0, n |= 0, o || Ae(this, t, e, n, Math.pow(2, 8 * n) - 1, 0);
  var d = n - 1, g = 1;
  for (this[e + d] = 255 & t; --d >= 0 && (g *= 256);) this[e + d] = t / g & 255;
  return e + n;
}, ut.prototype.writeUInt8 = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 1, 255, 0), ut.TYPED_ARRAY_SUPPORT || (t = Math.floor(t)), this[e] = 255 & t, e + 1;
}, ut.prototype.writeUInt16LE = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 2, 65535, 0), ut.TYPED_ARRAY_SUPPORT ? (this[e] = 255 & t, this[e + 1] = t >>> 8) : Wa(this, t, e, !0), e + 2;
}, ut.prototype.writeUInt16BE = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 2, 65535, 0), ut.TYPED_ARRAY_SUPPORT ? (this[e] = t >>> 8, this[e + 1] = 255 & t) : Wa(this, t, e, !1), e + 2;
}, ut.prototype.writeUInt32LE = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 4, 4294967295, 0), ut.TYPED_ARRAY_SUPPORT ? (this[e + 3] = t >>> 24, this[e + 2] = t >>> 16, this[e + 1] = t >>> 8, this[e] = 255 & t) : Xa(this, t, e, !0), e + 4;
}, ut.prototype.writeUInt32BE = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 4, 4294967295, 0), ut.TYPED_ARRAY_SUPPORT ? (this[e] = t >>> 24, this[e + 1] = t >>> 16, this[e + 2] = t >>> 8, this[e + 3] = 255 & t) : Xa(this, t, e, !1), e + 4;
}, ut.prototype.writeIntLE = function (t, e, n, o) {
  if (t = +t, e |= 0, !o) {
    var d = Math.pow(2, 8 * n - 1);
    Ae(this, t, e, n, d - 1, -d);
  }
  var g = 0, b = 1, _ = 0;
  for (this[e] = 255 & t; ++g < n && (b *= 256);) t < 0 && _ === 0 && this[e + g - 1] !== 0 && (_ = 1), this[e + g] = (t / b >> 0) - _ & 255;
  return e + n;
}, ut.prototype.writeIntBE = function (t, e, n, o) {
  if (t = +t, e |= 0, !o) {
    var d = Math.pow(2, 8 * n - 1);
    Ae(this, t, e, n, d - 1, -d);
  }
  var g = n - 1, b = 1, _ = 0;
  for (this[e + g] = 255 & t; --g >= 0 && (b *= 256);) t < 0 && _ === 0 && this[e + g + 1] !== 0 && (_ = 1), this[e + g] = (t / b >> 0) - _ & 255;
  return e + n;
}, ut.prototype.writeInt8 = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 1, 127, -128), ut.TYPED_ARRAY_SUPPORT || (t = Math.floor(t)), t < 0 && (t = 255 + t + 1), this[e] = 255 & t, e + 1;
}, ut.prototype.writeInt16LE = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 2, 32767, -32768), ut.TYPED_ARRAY_SUPPORT ? (this[e] = 255 & t, this[e + 1] = t >>> 8) : Wa(this, t, e, !0), e + 2;
}, ut.prototype.writeInt16BE = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 2, 32767, -32768), ut.TYPED_ARRAY_SUPPORT ? (this[e] = t >>> 8, this[e + 1] = 255 & t) : Wa(this, t, e, !1), e + 2;
}, ut.prototype.writeInt32LE = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 4, 2147483647, -2147483648), ut.TYPED_ARRAY_SUPPORT ? (this[e] = 255 & t, this[e + 1] = t >>> 8, this[e + 2] = t >>> 16, this[e + 3] = t >>> 24) : Xa(this, t, e, !0), e + 4;
}, ut.prototype.writeInt32BE = function (t, e, n) {
  return t = +t, e |= 0, n || Ae(this, t, e, 4, 2147483647, -2147483648), t < 0 && (t = 4294967295 + t + 1), ut.TYPED_ARRAY_SUPPORT ? (this[e] = t >>> 24, this[e + 1] = t >>> 16, this[e + 2] = t >>> 8, this[e + 3] = 255 & t) : Xa(this, t, e, !1), e + 4;
}, ut.prototype.writeFloatLE = function (t, e, n) {
  return Bh(this, t, e, !0, n);
}, ut.prototype.writeFloatBE = function (t, e, n) {
  return Bh(this, t, e, !1, n);
}, ut.prototype.writeDoubleLE = function (t, e, n) {
  return zh(this, t, e, !0, n);
}, ut.prototype.writeDoubleBE = function (t, e, n) {
  return zh(this, t, e, !1, n);
}, ut.prototype.copy = function (t, e, n, o) {
  if (n || (n = 0), o || o === 0 || (o = this.length), e >= t.length && (e = t.length), e || (e = 0), o > 0 && o < n && (o = n), o === n || t.length === 0 || this.length === 0) return 0;
  if (e < 0) throw new RangeError("targetStart out of bounds");
  if (n < 0 || n >= this.length) throw new RangeError("sourceStart out of bounds");
  if (o < 0) throw new RangeError("sourceEnd out of bounds");
  o > this.length && (o = this.length), t.length - e < o - n && (o = t.length - e + n);
  var d, g = o - n;
  if (this === t && n < e && e < o) for (d = g - 1; d >= 0; --d) t[d + e] = this[d + n];
  else if (g < 1e3 || !ut.TYPED_ARRAY_SUPPORT) for (d = 0; d < g; ++d) t[d + e] = this[d + n];
  else Uint8Array.prototype.set.call(t, this.subarray(n, n + g), e);
  return g;
}, ut.prototype.fill = function (t, e, n, o) {
  if (typeof t == "string") {
    if (typeof e == "string" ? (o = e, e = 0, n = this.length) : typeof n == "string" && (o = n, n = this.length), t.length === 1) {
      var d = t.charCodeAt(0);
      d < 256 && (t = d);
    }
    if (o !== void 0 && typeof o != "string") throw new TypeError("encoding must be a string");
    if (typeof o == "string" && !ut.isEncoding(o)) throw new TypeError("Unknown encoding: " + o);
  } else typeof t == "number" && (t &= 255);
  if (e < 0 || this.length < e || this.length < n) throw new RangeError("Out of range index");
  if (n <= e) return this;
  var g;
  if (e >>>= 0, n = n === void 0 ? this.length : n >>> 0, t || (t = 0), typeof t == "number") for (g = e; g < n; ++g) this[g] = t;
  else {
    var b = Tn(t) ? t : Lo(new ut(t, o).toString()), _ = b.length;
    for (g = 0; g < n - e; ++g) this[g + e] = b[g % _];
  }
  return this;
};
var Hv = /[^+\/0-9A-Za-z-_]/g;
function Wv(t) {
  return t < 16 ? "0" + t.toString(16) : t.toString(16);
}
function Lo(t, e) {
  var n;
  e = e || 1 / 0;
  for (var o = t.length, d = null, g = [], b = 0; b < o; ++b) {
    if ((n = t.charCodeAt(b)) > 55295 && n < 57344) {
      if (!d) {
        if (n > 56319) {
          (e -= 3) > -1 && g.push(239, 191, 189);
          continue;
        }
        if (b + 1 === o) {
          (e -= 3) > -1 && g.push(239, 191, 189);
          continue;
        }
        d = n;
        continue;
      }
      if (n < 56320) {
        (e -= 3) > -1 && g.push(239, 191, 189), d = n;
        continue;
      }
      n = 65536 + (d - 55296 << 10 | n - 56320);
    } else d && (e -= 3) > -1 && g.push(239, 191, 189);
    if (d = null, n < 128) {
      if ((e -= 1) < 0) break;
      g.push(n);
    } else if (n < 2048) {
      if ((e -= 2) < 0) break;
      g.push(n >> 6 | 192, 63 & n | 128);
    } else if (n < 65536) {
      if ((e -= 3) < 0) break;
      g.push(n >> 12 | 224, n >> 6 & 63 | 128, 63 & n | 128);
    } else {
      if (!(n < 1114112)) throw new Error("Invalid code point");
      if ((e -= 4) < 0) break;
      g.push(n >> 18 | 240, n >> 12 & 63 | 128, n >> 6 & 63 | 128, 63 & n | 128);
    }
  }
  return g;
}
function qp(t) {
  return function (e) {
    var n, o, d, g, b, _;
    Uc || zp();
    var w = e.length;
    if (w % 4 > 0) throw new Error("Invalid string. Length must be a multiple of 4");
    b = e[w - 2] === "=" ? 2 : e[w - 1] === "=" ? 1 : 0, _ = new Ov(3 * w / 4 - b), d = b > 0 ? w - 4 : w;
    var E = 0;
    for (n = 0, o = 0; n < d; n += 4, o += 3) g = qe[e.charCodeAt(n)] << 18 | qe[e.charCodeAt(n + 1)] << 12 | qe[e.charCodeAt(n + 2)] << 6 | qe[e.charCodeAt(n + 3)], _[E++] = g >> 16 & 255, _[E++] = g >> 8 & 255, _[E++] = 255 & g;
    return b === 2 ? (g = qe[e.charCodeAt(n)] << 2 | qe[e.charCodeAt(n + 1)] >> 4, _[E++] = 255 & g) : b === 1 && (g = qe[e.charCodeAt(n)] << 10 | qe[e.charCodeAt(n + 1)] << 4 | qe[e.charCodeAt(n + 2)] >> 2, _[E++] = g >> 8 & 255, _[E++] = 255 & g), _;
  }(function (e) {
    if ((e = function (n) {
      return n.trim ? n.trim() : n.replace(/^\s+|\s+$/g, "");
    }(e).replace(Hv, "")).length < 2) return "";
    for (; e.length % 4 != 0;) e += "=";
    return e;
  }(t));
}
function es(t, e, n, o) {
  for (var d = 0; d < o && !(d + n >= e.length || d >= t.length); ++d) e[d + n] = t[d];
  return d;
}
function Nh(t) {
  return !!t.constructor && typeof t.constructor.isBuffer == "function" && t.constructor.isBuffer(t);
}
function Xv(t) {
  if (t.__esModule) return t;
  var e = Object.defineProperty({}, "__esModule", { value: !0 });
  return Object.keys(t).forEach(function (n) {
    var o = Object.getOwnPropertyDescriptor(t, n);
    Object.defineProperty(e, n, o.get ? o : {
      enumerable: !0, get: function () {
        return t[n];
      }
    });
  }), e;
}
var Zp = {}, Uh = {}, Bs = Xv(Tv({ __proto__: null, default: Uh }, [Uh]));
(function (t) {
  /*! Fabric.js Copyright 2008-2015, Printio (Juriy Zaytsev, Maxim Chernyak) */
  var e, n, o, d, g, b, _, w, E, j, H, G, J, nt, rt, vt, Y, L, A, tt, dt, it, ht, v = v || { version: "4.6.0" };
  if (t.fabric = v, typeof document < "u" && typeof window < "u") document instanceof (typeof HTMLDocument < "u" ? HTMLDocument : Document) ? v.document = document : v.document = document.implementation.createHTMLDocument(""), v.window = window;
  else {
    var Ct = new Bs.JSDOM(decodeURIComponent("%3C!DOCTYPE%20html%3E%3Chtml%3E%3Chead%3E%3C%2Fhead%3E%3Cbody%3E%3C%2Fbody%3E%3C%2Fhtml%3E"), { features: { FetchExternalResources: ["img"] }, resources: "usable" }).window;
    v.document = Ct.document, v.jsdomImplForWrapper = Bs.implForWrapper, v.nodeCanvas = Bs.Canvas, v.window = Ct, DOMParser = v.window.DOMParser;
  }
  if (v.isTouchSupported = "ontouchstart" in v.window || "ontouchstart" in v.document || v.window && v.window.navigator && v.window.navigator.maxTouchPoints > 0, v.isLikelyNode = ut !== void 0 && typeof window > "u", v.SHARED_ATTRIBUTES = ["display", "transform", "fill", "fill-opacity", "fill-rule", "opacity", "stroke", "stroke-dasharray", "stroke-linecap", "stroke-dashoffset", "stroke-linejoin", "stroke-miterlimit", "stroke-opacity", "stroke-width", "id", "paint-order", "vector-effect", "instantiated_by_use", "clip-path"], v.DPI = 96, v.reNum = "(?:[-+]?(?:\\d+|\\d*\\.\\d+)(?:[eE][-+]?\\d+)?)", v.commaWsp = "(?:\\s+,?\\s*|,\\s*)", v.rePathCommand = /([-+]?((\d+\.\d+)|((\d+)|(\.\d+)))(?:[eE][-+]?\d+)?)/gi, v.reNonWord = /[ \n\.,;!\?\-]/, v.fontPaths = {}, v.iMatrix = [1, 0, 0, 1, 0, 0], v.svgNS = "http://www.w3.org/2000/svg", v.perfLimitSizeTotal = 2097152, v.maxCacheSideLimit = 4096, v.minCacheSideLimit = 256, v.charWidthsCache = {}, v.textureSize = 2048, v.disableStyleCopyPaste = !1, v.enableGLFiltering = !0, v.devicePixelRatio = v.window.devicePixelRatio || v.window.webkitDevicePixelRatio || v.window.mozDevicePixelRatio || 1, v.browserShadowBlurConstant = 1, v.arcToSegmentsCache = {}, v.boundsOfCurveCache = {}, v.cachesBoundsOfCurve = !0, v.forceGLPutImageData = !1, v.initFilterBackend = function () {
    return v.enableGLFiltering && v.isWebglSupported && v.isWebglSupported(v.textureSize) ? (console.log("max texture size: " + v.maxTextureSize), new v.WebglFilterBackend({ tileSize: v.textureSize })) : v.Canvas2dFilterBackend ? new v.Canvas2dFilterBackend() : void 0;
  }, typeof document < "u" && typeof window < "u" && (window.fabric = v), F === void 0) var F = {};
  if (function (l) {
    l.modifyEventListener = !1, l.modifySelectors = !1, l.configure = function (C) {
      isFinite(C.modifyEventListener) && (l.modifyEventListener = C.modifyEventListener), isFinite(C.modifySelectors) && (l.modifySelectors = C.modifySelectors), S === !1 && l.modifyEventListener && k(), D === !1 && l.modifySelectors && M();
    }, l.add = function (C, O, N, W) {
      return c(C, O, N, W, "add");
    }, l.remove = function (C, O, N, W) {
      return c(C, O, N, W, "remove");
    }, l.returnFalse = function (C) {
      return !1;
    }, l.stop = function (C) {
      C && (C.stopPropagation && C.stopPropagation(), C.cancelBubble = !0, C.cancelBubbleCount = 0);
    }, l.prevent = function (C) {
      C && (C.preventDefault ? C.preventDefault() : C.preventManipulation ? C.preventManipulation() : C.returnValue = !1);
    }, l.cancel = function (C) {
      l.stop(C), l.prevent(C);
    }, l.blur = function () {
      var C = document.activeElement;
      if (C) {
        var O = document.activeElement.nodeName;
        O !== "INPUT" && O !== "TEXTAREA" && C.contentEditable !== "true" || C.blur && C.blur();
      }
    }, l.getEventSupport = function (C, O) {
      if (typeof C == "string" && (O = C, C = window), (O = "on" + O) in C) return !0;
      if (C.setAttribute || (C = document.createElement("div")), C.setAttribute && C.removeAttribute) {
        C.setAttribute(O, "");
        var N = typeof C[O] == "function";
        return C[O] !== void 0 && (C[O] = null), C.removeAttribute(O), N;
      }
    };
    var a = function (C) {
      if (!C || typeof C != "object") return C;
      var O = new C.constructor();
      for (var N in C) C[N] && typeof C[N] == "object" ? O[N] = a(C[N]) : O[N] = C[N];
      return O;
    }, c = function (C, O, N, W, X, I) {
      if (W = W || {}, String(C) === "[object Object]") {
        var T = C;
        if (C = T.target, delete T.target, !T.type || !T.listener) {
          for (var R in T) {
            var Q = T[R];
            typeof Q != "function" && (W[R] = Q);
          }
          var K = {};
          for (var U in T) {
            R = U.split(",");
            var B = T[U], Z = {};
            for (var z in W) Z[z] = W[z];
            if (typeof B == "function") N = B;
            else {
              if (typeof B.listener != "function") continue;
              N = B.listener;
              for (var z in B) typeof B[z] != "function" && (Z[z] = B[z]);
            }
            for (var V = 0; V < R.length; V++) K[U] = F.add(C, R[V], N, Z, X);
          }
          return K;
        }
        for (var U in O = T.type, delete T.type, N = T.listener, delete T.listener, T) W[U] = T[U];
      }
      if (C && O && N) {
        if (typeof C == "string" && O === "ready") {
          if (!window.eventjs_stallOnReady) {
            var et = (/* @__PURE__ */ new Date()).getTime(), lt = W.timeout, at = W.interval || 1e3 / 60, ft = window.setInterval(function () {
              (/* @__PURE__ */ new Date()).getTime() - et > lt && window.clearInterval(ft), document.querySelector(C) && (window.clearInterval(ft), setTimeout(N, 1));
            }, at);
            return;
          }
          O = "load", C = window;
        }
        if (typeof C == "string") {
          if ((C = document.querySelectorAll(C)).length === 0) return i("Missing target on listener!", arguments);
          C.length === 1 && (C = C[0]);
        }
        var ot, ct = {};
        if (C.length > 0 && C !== window) {
          for (var st = 0, yt = C.length; st < yt; st++) (ot = c(C[st], O, N, a(W), X)) && (ct[st] = ot);
          return f(ct);
        }
        if (typeof O == "string" && ((O = O.toLowerCase()).indexOf(" ") !== -1 ? O = O.split(" ") : O.indexOf(",") !== -1 && (O = O.split(","))), typeof O != "string") {
          if (typeof O.length == "number") for (var gt = 0, Tt = O.length; gt < Tt; gt++) (ot = c(C, O[gt], N, a(W), X)) && (ct[O[gt]] = ot);
          else for (var U in O) (ot = typeof O[U] == "function" ? c(C, U, O[U], a(W), X) : c(C, U, O[U].listener, a(O[U]), X)) && (ct[U] = ot);
          return f(ct);
        }
        if (O.indexOf("on") === 0 && (O = O.substr(2)), typeof C != "object") return i("Target is not defined!", arguments);
        if (typeof N != "function") return i("Listener is not a function!", arguments);
        var kt = W.useCapture || !1, jt = m(C) + "." + m(N) + "." + (kt ? 1 : 0);
        if (l.Gesture && l.Gesture._gestureHandlers[O]) {
          if (jt = O + jt, X === "remove") {
            if (!h[jt]) return;
            h[jt].remove(), delete h[jt];
          } else if (X === "add") {
            if (h[jt]) return h[jt].add(), h[jt];
            if (W.useCall && !l.modifyEventListener) {
              var Kt = N;
              N = function (se, zt) {
                for (var ie in zt) se[ie] = zt[ie];
                return Kt.call(C, se);
              };
            }
            W.gesture = O, W.target = C, W.listener = N, W.fromOverwrite = I, h[jt] = l.proxy[O](W);
          }
          return h[jt];
        }
        var Rt, Bt = u(O);
        for (V = 0; V < Bt.length; V++) if (Rt = (O = Bt[V]) + "." + jt, X === "remove") {
          if (!h[Rt]) continue;
          C[x](O, N, kt), delete h[Rt];
        } else if (X === "add") {
          if (h[Rt]) return h[Rt];
          C[y](O, N, kt), h[Rt] = {
            id: Rt, type: O, target: C, listener: N, remove: function () {
              for (var se = 0; se < Bt.length; se++) l.remove(C, Bt[se], N, W);
            }
          };
        }
        return h[Rt];
      }
    }, f = function (C) {
      return {
        remove: function () {
          for (var O in C) C[O].remove();
        }, add: function () {
          for (var O in C) C[O].add();
        }
      };
    }, i = function (C, O) {
      typeof console < "u" && console.error !== void 0 && console.error(C, O);
    }, r = { msPointer: ["MSPointerDown", "MSPointerMove", "MSPointerUp"], touch: ["touchstart", "touchmove", "touchend"], mouse: ["mousedown", "mousemove", "mouseup"] }, s = { MSPointerDown: 0, MSPointerMove: 1, MSPointerUp: 2, touchstart: 0, touchmove: 1, touchend: 2, mousedown: 0, mousemove: 1, mouseup: 2 };
    l.supports = {}, window.navigator.msPointerEnabled && (l.supports.msPointer = !0), l.getEventSupport("touchstart") && (l.supports.touch = !0), l.getEventSupport("mousedown") && (l.supports.mouse = !0);
    var u = function (C) {
      var O = document.addEventListener ? "" : "on", N = s[C];
      if (isFinite(N)) {
        var W = [];
        for (var X in l.supports) W.push(O + r[X][N]);
        return W;
      }
      return [O + C];
    }, h = {}, p = 0, m = function (C) {
      return C === window ? "#window" : C === document ? "#document" : (C.uniqueID || (C.uniqueID = "e" + p++), C.uniqueID);
    }, y = document.addEventListener ? "addEventListener" : "attachEvent", x = document.removeEventListener ? "removeEventListener" : "detachEvent";
    l.createPointerEvent = function (C, O, N) {
      var W = O.gesture, X = O.target, I = C.changedTouches || l.proxy.getCoords(C);
      if (I.length) {
        var T = I[0];
        O.pointers = N ? [] : I, O.pageX = T.pageX, O.pageY = T.pageY, O.x = O.pageX, O.y = O.pageY;
      }
      var R = document.createEvent("Event");
      for (var Q in R.initEvent(W, !0, !0), R.originalEvent = C, O) Q !== "target" && (R[Q] = O[Q]);
      var K = R.type;
      l.Gesture && l.Gesture._gestureHandlers[K] && O.oldListener.call(X, R, O, !1);
    };
    var S = !1, k = function () {
      if (window.HTMLElement) {
        var C = function (O) {
          var N = function (W) {
            var X = W + "EventListener", I = O[X];
            O[X] = function (T, R, Q) {
              if (l.Gesture && l.Gesture._gestureHandlers[T]) {
                var K = Q;
                typeof Q == "object" ? K.useCall = !0 : K = { useCall: !0, useCapture: Q }, c(this, T, R, K, W, !0);
              } else for (var U = u(T), B = 0; B < U.length; B++) I.call(this, U[B], R, Q);
            };
          };
          N("add"), N("remove");
        };
        navigator.userAgent.match(/Firefox/) ? (C(HTMLDivElement.prototype), C(HTMLCanvasElement.prototype)) : C(HTMLElement.prototype), C(document), C(window);
      }
    }, D = !1, M = function () {
      var C = NodeList.prototype;
      C.removeEventListener = function (O, N, W) {
        for (var X = 0, I = this.length; X < I; X++) this[X].removeEventListener(O, N, W);
      }, C.addEventListener = function (O, N, W) {
        for (var X = 0, I = this.length; X < I; X++) this[X].addEventListener(O, N, W);
      };
    };
  }(F), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = function (l) {
    l.pointerSetup = function (s, u) {
      s.target = s.target || window, s.doc = s.target.ownerDocument || s.target, s.minFingers = s.minFingers || s.fingers || 1, s.maxFingers = s.maxFingers || s.fingers || 1 / 0, s.position = s.position || "relative", delete s.fingers, (u = u || {}).enabled = !0, u.gesture = s.gesture, u.target = s.target, u.env = s.env, F.modifyEventListener && s.fromOverwrite && (s.oldListener = s.listener, s.listener = F.createPointerEvent);
      var h = 0, p = u.gesture.indexOf("pointer") === 0 && F.modifyEventListener ? "pointer" : "mouse";
      return s.oldListener && (u.oldListener = s.oldListener), u.listener = s.listener, u.proxy = function (m) {
        u.defaultListener = s.listener, s.listener = m, m(s.event, u);
      }, u.add = function () {
        u.enabled !== !0 && (s.onPointerDown && F.add(s.target, p + "down", s.onPointerDown), s.onPointerMove && F.add(s.doc, p + "move", s.onPointerMove), s.onPointerUp && F.add(s.doc, p + "up", s.onPointerUp), u.enabled = !0);
      }, u.remove = function () {
        u.enabled !== !1 && (s.onPointerDown && F.remove(s.target, p + "down", s.onPointerDown), s.onPointerMove && F.remove(s.doc, p + "move", s.onPointerMove), s.onPointerUp && F.remove(s.doc, p + "up", s.onPointerUp), u.reset(), u.enabled = !1);
      }, u.pause = function (m) {
        !s.onPointerMove || m && !m.move || F.remove(s.doc, p + "move", s.onPointerMove), !s.onPointerUp || m && !m.up || F.remove(s.doc, p + "up", s.onPointerUp), h = s.fingers, s.fingers = 0;
      }, u.resume = function (m) {
        !s.onPointerMove || m && !m.move || F.add(s.doc, p + "move", s.onPointerMove), !s.onPointerUp || m && !m.up || F.add(s.doc, p + "up", s.onPointerUp), s.fingers = h;
      }, u.reset = function () {
        s.tracker = {}, s.fingers = 0;
      }, u;
    };
    var a = F.supports;
    F.isMouse = !!a.mouse, F.isMSPointer = !!a.touch, F.isTouch = !!a.msPointer, l.pointerStart = function (s, u, h) {
      var p = (s.type || "mousedown").toUpperCase();
      p.indexOf("MOUSE") === 0 ? (F.isMouse = !0, F.isTouch = !1, F.isMSPointer = !1) : p.indexOf("TOUCH") === 0 ? (F.isMouse = !1, F.isTouch = !0, F.isMSPointer = !1) : p.indexOf("MSPOINTER") === 0 && (F.isMouse = !1, F.isTouch = !1, F.isMSPointer = !0);
      var m = function (W, X) {
        var I = h.bbox, T = x[X] = {};
        switch (h.position) {
          case "absolute":
            T.offsetX = 0, T.offsetY = 0;
            break;
          case "differenceFromLast":
          case "difference":
            T.offsetX = W.pageX, T.offsetY = W.pageY;
            break;
          case "move":
            T.offsetX = W.pageX - I.x1, T.offsetY = W.pageY - I.y1;
            break;
          default:
            T.offsetX = I.x1 - I.scrollLeft, T.offsetY = I.y1 - I.scrollTop;
        }
        var R = W.pageX - T.offsetX, Q = W.pageY - T.offsetY;
        T.rotation = 0, T.scale = 1, T.startTime = T.moveTime = (/* @__PURE__ */ new Date()).getTime(), T.move = { x: R, y: Q }, T.start = { x: R, y: Q }, h.fingers++;
      };
      h.event = s, u.defaultListener && (h.listener = u.defaultListener, delete u.defaultListener);
      for (var y = !h.fingers, x = h.tracker, S = s.changedTouches || l.getCoords(s), k = S.length, D = 0; D < k; D++) {
        var M = S[D], C = M.identifier || 1 / 0;
        if (h.fingers) {
          if (h.fingers >= h.maxFingers) {
            var O = [];
            for (var C in h.tracker) O.push(C);
            return u.identifier = O.join(","), y;
          }
          for (var N in x) if (x[N].up) {
            delete x[N], m(M, C), h.cancel = !0;
            break;
          }
          if (x[C]) continue;
          m(M, C);
        } else x = h.tracker = {}, u.bbox = h.bbox = l.getBoundingBox(h.target), h.fingers = 0, h.cancel = !1, m(M, C);
      }
      O = [];
      for (var C in h.tracker) O.push(C);
      return u.identifier = O.join(","), y;
    }, l.pointerEnd = function (s, u, h, p) {
      for (var m = s.touches || [], y = m.length, x = {}, S = 0; S < y; S++)
        x[(k = m[S].identifier) || 1 / 0] = !0;
      for (var k in h.tracker) {
        var D = h.tracker[k];
        x[k] || D.up || (p && p({ pageX: D.pageX, pageY: D.pageY, changedTouches: [{ pageX: D.pageX, pageY: D.pageY, identifier: k === "Infinity" ? 1 / 0 : k }] }, "up"), D.up = !0, h.fingers--);
      }
      if (h.fingers !== 0) return !1;
      var M = [];
      for (var k in h.gestureFingers = 0, h.tracker) h.gestureFingers++, M.push(k);
      return u.identifier = M.join(","), !0;
    }, l.getCoords = function (s) {
      return s.pageX !== void 0 ? l.getCoords = function (u) {
        return Array({ type: "mouse", x: u.pageX, y: u.pageY, pageX: u.pageX, pageY: u.pageY, identifier: u.pointerId || 1 / 0 });
      } : l.getCoords = function (u) {
        var h = document.documentElement;
        return u = u || window.event, Array({ type: "mouse", x: u.clientX + h.scrollLeft, y: u.clientY + h.scrollTop, pageX: u.clientX + h.scrollLeft, pageY: u.clientY + h.scrollTop, identifier: 1 / 0 });
      }, l.getCoords(s);
    }, l.getCoord = function (s) {
      if ("ontouchstart" in window) {
        var u = 0, h = 0;
        l.getCoord = function (p) {
          var m = p.changedTouches;
          return m && m.length ? { x: u = m[0].pageX, y: h = m[0].pageY } : { x: u, y: h };
        };
      } else s.pageX !== void 0 && s.pageY !== void 0 ? l.getCoord = function (p) {
        return { x: p.pageX, y: p.pageY };
      } : l.getCoord = function (p) {
        var m = document.documentElement;
        return { x: (p = p || window.event).clientX + m.scrollLeft, y: p.clientY + m.scrollTop };
      };
      return l.getCoord(s);
    };
    var c, f, i, r = function (s, u) {
      var h = parseFloat(s.getPropertyValue(u), 10);
      return isFinite(h) ? h : 0;
    };
    return l.getBoundingBox = function (s) {
      s !== window && s !== document || (s = document.body);
      var u = {}, h = s.getBoundingClientRect();
      if (u.width = h.width, u.height = h.height, u.x1 = h.left, u.y1 = h.top, u.scaleX = h.width / s.offsetWidth || 1, u.scaleY = h.height / s.offsetHeight || 1, u.scrollLeft = 0, u.scrollTop = 0, (k = window.getComputedStyle(s)).getPropertyValue("box-sizing") !== "border-box") {
        var p = r(k, "border-left-width"), m = r(k, "border-right-width"), y = r(k, "border-bottom-width"), x = r(k, "border-top-width");
        u.border = [p, m, x, y], u.x1 += p, u.y1 += x, u.width -= m + p, u.height -= y + x;
      }
      u.x2 = u.x1 + u.width, u.y2 = u.y1 + u.height;
      for (var S = (D = k.getPropertyValue("position")) === "fixed" ? s : s.parentNode; S !== null && S !== document.body && S.scrollTop !== void 0;) {
        var k, D;
        if ((D = (k = window.getComputedStyle(S)).getPropertyValue("position")) !== "absolute") {
          if (D === "fixed") {
            u.scrollTop -= S.parentNode.scrollTop, u.scrollLeft -= S.parentNode.scrollLeft;
            break;
          }
          u.scrollLeft += S.scrollLeft, u.scrollTop += S.scrollTop;
        }
        S = S.parentNode;
      }
      return u.scrollBodyLeft = window.pageXOffset !== void 0 ? window.pageXOffset : (document.documentElement || document.body.parentNode || document.body).scrollLeft, u.scrollBodyTop = window.pageYOffset !== void 0 ? window.pageYOffset : (document.documentElement || document.body.parentNode || document.body).scrollTop, u.scrollLeft -= u.scrollBodyLeft, u.scrollTop -= u.scrollBodyTop, u;
    }, f = navigator.userAgent.toLowerCase(), i = f.indexOf("macintosh") !== -1, c = i && f.indexOf("khtml") !== -1 ? { 91: !0, 93: !0 } : i && f.indexOf("firefox") !== -1 ? { 224: !0 } : { 17: !0 }, (l.metaTrackerReset = function () {
      F.fnKey = l.fnKey = !1, F.metaKey = l.metaKey = !1, F.escKey = l.escKey = !1, F.ctrlKey = l.ctrlKey = !1, F.shiftKey = l.shiftKey = !1, F.altKey = l.altKey = !1;
    })(), l.metaTracker = function (s) {
      var u = s.type === "keydown";
      s.keyCode === 27 && (F.escKey = l.escKey = u), c[s.keyCode] && (F.metaKey = l.metaKey = u), F.ctrlKey = l.ctrlKey = s.ctrlKey, F.shiftKey = l.shiftKey = s.shiftKey, F.altKey = l.altKey = s.altKey;
    }, l;
  }(F.proxy), F === void 0 && (F = {}), F.MutationObserver = (e = window.MutationObserver || window.WebKitMutationObserver || window.MozMutationObserver, n = !e && function () {
    var l = document.createElement("p"), a = !1, c = function () {
      a = !0;
    };
    if (l.addEventListener) l.addEventListener("DOMAttrModified", c, !1);
    else {
      if (!l.attachEvent) return !1;
      l.attachEvent("onDOMAttrModified", c);
    }
    return l.setAttribute("id", "target"), a;
  }(), function (l, a) {
    e ? new e(function (c) {
      c.forEach(function (f) {
        a.call(f.target, f.attributeName);
      });
    }).observe(l, { subtree: !1, attributes: !0 }) : n ? F.add(l, "DOMAttrModified", function (c) {
      a.call(l, c.attrName);
    }) : "onpropertychange" in document.body && F.add(l, "propertychange", function (c) {
      a.call(l, window.event.propertyName);
    });
  }), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = ((o = F.proxy).click = function (l) {
    l.gesture = l.gesture || "click", l.maxFingers = l.maxFingers || l.fingers || 1, l.onPointerDown = function (c) {
      o.pointerStart(c, a, l) && F.add(l.target, "mouseup", l.onPointerUp);
    }, l.onPointerUp = function (c) {
      if (o.pointerEnd(c, a, l)) {
        F.remove(l.target, "mouseup", l.onPointerUp);
        var f = (c.changedTouches || o.getCoords(c))[0], i = l.bbox, r = o.getBoundingBox(l.target), s = f.pageY - r.scrollBodyTop, u = f.pageX - r.scrollBodyLeft;
        if (u > i.x1 && s > i.y1 && u < i.x2 && s < i.y2 && i.scrollTop === r.scrollTop) {
          for (var h in l.tracker) break;
          var p = l.tracker[h];
          a.x = p.start.x, a.y = p.start.y, l.listener(c, a);
        }
      }
    };
    var a = o.pointerSetup(l);
    return a.state = "click", F.add(l.target, "mousedown", l.onPointerDown), a;
  }, F.Gesture = F.Gesture || {}, F.Gesture._gestureHandlers = F.Gesture._gestureHandlers || {}, F.Gesture._gestureHandlers.click = o.click, o), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = function (l) {
    return l.dbltap = l.dblclick = function (a) {
      a.gesture = a.gesture || "dbltap", a.maxFingers = a.maxFingers || a.fingers || 1;
      var c, f, i, r, s;
      a.onPointerDown = function (h) {
        var p = h.changedTouches || l.getCoords(h);
        c && !f ? (s = p[0], f = (/* @__PURE__ */ new Date()).getTime() - c) : (r = p[0], c = (/* @__PURE__ */ new Date()).getTime(), f = 0, clearTimeout(i), i = setTimeout(function () {
          c = 0;
        }, 700)), l.pointerStart(h, u, a) && (F.add(a.target, "mousemove", a.onPointerMove).listener(h), F.add(a.target, "mouseup", a.onPointerUp));
      }, a.onPointerMove = function (h) {
        if (c && !f) {
          var p = h.changedTouches || l.getCoords(h);
          s = p[0];
        }
        var m = a.bbox, y = s.pageX - m.x1, x = s.pageY - m.y1;
        y > 0 && y < m.width && x > 0 && x < m.height && Math.abs(s.pageX - r.pageX) <= 25 && Math.abs(s.pageY - r.pageY) <= 25 || (F.remove(a.target, "mousemove", a.onPointerMove), clearTimeout(i), c = f = 0);
      }, a.onPointerUp = function (h) {
        if (l.pointerEnd(h, u, a) && (F.remove(a.target, "mousemove", a.onPointerMove), F.remove(a.target, "mouseup", a.onPointerUp)), c && f) {
          if (f <= 700) {
            for (var p in u.state = a.gesture, a.tracker) break;
            var m = a.tracker[p];
            u.x = m.start.x, u.y = m.start.y, a.listener(h, u);
          }
          clearTimeout(i), c = f = 0;
        }
      };
      var u = l.pointerSetup(a);
      return u.state = "dblclick", F.add(a.target, "mousedown", a.onPointerDown), u;
    }, F.Gesture = F.Gesture || {}, F.Gesture._gestureHandlers = F.Gesture._gestureHandlers || {}, F.Gesture._gestureHandlers.dbltap = l.dbltap, F.Gesture._gestureHandlers.dblclick = l.dblclick, l;
  }(F.proxy), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = function (l) {
    return l.dragElement = function (a, c) {
      l.drag({
        event: c, target: a, position: "move", listener: function (f, i) {
          a.style.left = i.x + "px", a.style.top = i.y + "px", F.prevent(f);
        }
      });
    }, l.drag = function (a) {
      a.gesture = "drag", a.onPointerDown = function (f) {
        l.pointerStart(f, c, a) && (a.monitor || (F.add(a.doc, "mousemove", a.onPointerMove), F.add(a.doc, "mouseup", a.onPointerUp))), a.onPointerMove(f, "down");
      }, a.onPointerMove = function (f, i) {
        if (!a.tracker) return a.onPointerDown(f);
        a.bbox;
        for (var r = f.changedTouches || l.getCoords(f), s = r.length, u = 0; u < s; u++) {
          var h = r[u], p = h.identifier || 1 / 0, m = a.tracker[p];
          m && (m.pageX = h.pageX, m.pageY = h.pageY, c.state = i || "move", c.identifier = p, c.start = m.start, c.fingers = a.fingers, a.position === "differenceFromLast" ? (c.x = m.pageX - m.offsetX, c.y = m.pageY - m.offsetY, m.offsetX = m.pageX, m.offsetY = m.pageY) : (c.x = m.pageX - m.offsetX, c.y = m.pageY - m.offsetY), a.listener(f, c));
        }
      }, a.onPointerUp = function (f) {
        l.pointerEnd(f, c, a, a.onPointerMove) && (a.monitor || (F.remove(a.doc, "mousemove", a.onPointerMove), F.remove(a.doc, "mouseup", a.onPointerUp)));
      };
      var c = l.pointerSetup(a);
      return a.event ? a.onPointerDown(a.event) : (F.add(a.target, "mousedown", a.onPointerDown), a.monitor && (F.add(a.doc, "mousemove", a.onPointerMove), F.add(a.doc, "mouseup", a.onPointerUp))), c;
    }, F.Gesture = F.Gesture || {}, F.Gesture._gestureHandlers = F.Gesture._gestureHandlers || {}, F.Gesture._gestureHandlers.drag = l.drag, l;
  }(F.proxy), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = function (l) {
    var a = Math.PI / 180, c = function (f, i) {
      var r = 0, s = 0, u = 0;
      for (var h in i) {
        var p = i[h];
        p.up || (r += p.move.x, s += p.move.y, u++);
      }
      return f.x = r /= u, f.y = s /= u, f;
    };
    return l.gesture = function (f) {
      f.gesture = f.gesture || "gesture", f.minFingers = f.minFingers || f.fingers || 2, f.onPointerDown = function (r) {
        var s = f.fingers;
        if (l.pointerStart(r, i, f) && (F.add(f.doc, "mousemove", f.onPointerMove), F.add(f.doc, "mouseup", f.onPointerUp)), f.fingers === f.minFingers && s !== f.fingers) {
          i.fingers = f.minFingers, i.scale = 1, i.rotation = 0, i.state = "start";
          var u = "";
          for (var h in f.tracker) u += h;
          i.identifier = parseInt(u), c(i, f.tracker), f.listener(r, i);
        }
      }, f.onPointerMove = function (r, s) {
        for (var u = f.bbox, h = f.tracker, p = (x = r.changedTouches || l.getCoords(r)).length, m = 0; m < p; m++) {
          var y = h[D = (M = x[m]).identifier || 1 / 0];
          y && (y.move.x = M.pageX - u.x1, y.move.y = M.pageY - u.y1);
        }
        if (!(f.fingers < f.minFingers)) {
          var x = [], S = 0, k = 0;
          for (var D in c(i, h), h) {
            var M;
            if (!(M = h[D]).up) {
              var C = M.start;
              if (!C.distance) {
                var O = C.x - i.x, N = C.y - i.y;
                C.distance = Math.sqrt(O * O + N * N), C.angle = Math.atan2(O, N) / a;
              }
              O = M.move.x - i.x, N = M.move.y - i.y, S += Math.sqrt(O * O + N * N) / C.distance;
              var W = Math.atan2(O, N) / a, X = (C.angle - W + 360) % 360 - 180;
              M.DEG2 = M.DEG1, M.DEG1 = X > 0 ? X : -X, M.DEG2 !== void 0 && (X > 0 ? M.rotation += M.DEG1 - M.DEG2 : M.rotation -= M.DEG1 - M.DEG2, k += M.rotation), x.push(M.move);
            }
          }
          i.touches = x, i.fingers = f.fingers, i.scale = S / f.fingers, i.rotation = k / f.fingers, i.state = "change", f.listener(r, i);
        }
      }, f.onPointerUp = function (r) {
        var s = f.fingers;
        l.pointerEnd(r, i, f) && (F.remove(f.doc, "mousemove", f.onPointerMove), F.remove(f.doc, "mouseup", f.onPointerUp)), s === f.minFingers && f.fingers < f.minFingers && (i.fingers = f.fingers, i.state = "end", f.listener(r, i));
      };
      var i = l.pointerSetup(f);
      return F.add(f.target, "mousedown", f.onPointerDown), i;
    }, F.Gesture = F.Gesture || {}, F.Gesture._gestureHandlers = F.Gesture._gestureHandlers || {}, F.Gesture._gestureHandlers.gesture = l.gesture, l;
  }(F.proxy), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = function (l) {
    return l.pointerdown = l.pointermove = l.pointerup = function (a) {
      if (a.gesture = a.gesture || "pointer", !a.target.isPointerEmitter) {
        var c = !0;
        a.onPointerDown = function (i) {
          c = !1, f.gesture = "pointerdown", a.listener(i, f);
        }, a.onPointerMove = function (i) {
          f.gesture = "pointermove", a.listener(i, f, c);
        }, a.onPointerUp = function (i) {
          c = !0, f.gesture = "pointerup", a.listener(i, f, !0);
        };
        var f = l.pointerSetup(a);
        return F.add(a.target, "mousedown", a.onPointerDown), F.add(a.target, "mousemove", a.onPointerMove), F.add(a.doc, "mouseup", a.onPointerUp), a.target.isPointerEmitter = !0, f;
      }
    }, F.Gesture = F.Gesture || {}, F.Gesture._gestureHandlers = F.Gesture._gestureHandlers || {}, F.Gesture._gestureHandlers.pointerdown = l.pointerdown, F.Gesture._gestureHandlers.pointermove = l.pointermove, F.Gesture._gestureHandlers.pointerup = l.pointerup, l;
  }(F.proxy), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = function (l) {
    return l.shake = function (a) {
      var c = {
        gesture: "devicemotion", acceleration: {}, accelerationIncludingGravity: {}, target: a.target, listener: a.listener, remove: function () {
          window.removeEventListener("devicemotion", s, !1);
        }
      }, f = (/* @__PURE__ */ new Date()).getTime(), i = { x: 0, y: 0, z: 0 }, r = { x: { count: 0, value: 0 }, y: { count: 0, value: 0 }, z: { count: 0, value: 0 } }, s = function (u) {
        var h = 0.8, p = u.accelerationIncludingGravity;
        if (i.x = h * i.x + (1 - h) * p.x, i.y = h * i.y + (1 - h) * p.y, i.z = h * i.z + (1 - h) * p.z, c.accelerationIncludingGravity = i, c.acceleration.x = p.x - i.x, c.acceleration.y = p.y - i.y, c.acceleration.z = p.z - i.z, a.gesture !== "devicemotion") for (var m = (/* @__PURE__ */ new Date()).getTime(), y = 0; y < 3; y++) {
          var x = "xyz"[y], S = c.acceleration[x], k = r[x], D = Math.abs(S);
          if (!(m - f < 1e3) && D > 4) {
            var M = m * S / D, C = Math.abs(M + k.value);
            k.value && C < 200 ? (k.value = M, k.count++, k.count === 3 && (a.listener(u, c), f = m, k.value = 0, k.count = 0)) : (k.value = M, k.count = 1);
          }
        }
        else a.listener(u, c);
      };
      if (window.addEventListener) return window.addEventListener("devicemotion", s, !1), c;
    }, F.Gesture = F.Gesture || {}, F.Gesture._gestureHandlers = F.Gesture._gestureHandlers || {}, F.Gesture._gestureHandlers.shake = l.shake, l;
  }(F.proxy), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = function (l) {
    var a = Math.PI / 180;
    return l.swipe = function (c) {
      c.snap = c.snap || 90, c.threshold = c.threshold || 1, c.gesture = c.gesture || "swipe", c.onPointerDown = function (i) {
        l.pointerStart(i, f, c) && (F.add(c.doc, "mousemove", c.onPointerMove).listener(i), F.add(c.doc, "mouseup", c.onPointerUp));
      }, c.onPointerMove = function (i) {
        for (var r = i.changedTouches || l.getCoords(i), s = r.length, u = 0; u < s; u++) {
          var h = r[u], p = h.identifier || 1 / 0, m = c.tracker[p];
          m && (m.move.x = h.pageX, m.move.y = h.pageY, m.moveTime = (/* @__PURE__ */ new Date()).getTime());
        }
      }, c.onPointerUp = function (i) {
        if (l.pointerEnd(i, f, c)) {
          var r, s;
          F.remove(c.doc, "mousemove", c.onPointerMove), F.remove(c.doc, "mouseup", c.onPointerUp);
          var u = { x: 0, y: 0 }, h = 0, p = 0, m = 0;
          for (var y in c.tracker) {
            var x = c.tracker[y], S = x.move.x - x.start.x, k = x.move.y - x.start.y;
            h += x.move.x, p += x.move.y, u.x += x.start.x, u.y += x.start.y, m++;
            var D = Math.sqrt(S * S + k * k), M = x.moveTime - x.startTime, C = Math.atan2(S, k) / a + 180, O = M ? D / M : 0;
            if (s === void 0) s = C, r = O;
            else {
              if (!(Math.abs(C - s) <= 20)) return;
              s = (s + C) / 2, r = (r + O) / 2;
            }
          }
          var N = c.gestureFingers;
          c.minFingers <= N && c.maxFingers >= N && r > c.threshold && (u.x /= m, u.y /= m, f.start = u, f.x = h / m, f.y = p / m, f.angle = -(((s / c.snap + 0.5 >> 0) * c.snap || 360) - 360), f.velocity = r, f.fingers = N, f.state = "swipe", c.listener(i, f));
        }
      };
      var f = l.pointerSetup(c);
      return F.add(c.target, "mousedown", c.onPointerDown), f;
    }, F.Gesture = F.Gesture || {}, F.Gesture._gestureHandlers = F.Gesture._gestureHandlers || {}, F.Gesture._gestureHandlers.swipe = l.swipe, l;
  }(F.proxy), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = function (l) {
    return l.longpress = function (a) {
      return a.gesture = "longpress", l.tap(a);
    }, l.tap = function (a) {
      var c, f;
      a.delay = a.delay || 500, a.timeout = a.timeout || 250, a.driftDeviance = a.driftDeviance || 10, a.gesture = a.gesture || "tap", a.onPointerDown = function (r) {
        if (l.pointerStart(r, i, a)) {
          if (c = (/* @__PURE__ */ new Date()).getTime(), F.add(a.doc, "mousemove", a.onPointerMove).listener(r), F.add(a.doc, "mouseup", a.onPointerUp), a.gesture !== "longpress") return;
          f = setTimeout(function () {
            if (!(r.cancelBubble && ++r.cancelBubbleCount > 1)) {
              var s = 0;
              for (var u in a.tracker) {
                var h = a.tracker[u];
                if (h.end === !0 || a.cancel) return;
                s++;
              }
              a.minFingers <= s && a.maxFingers >= s && (i.state = "start", i.fingers = s, i.x = h.start.x, i.y = h.start.y, a.listener(r, i));
            }
          }, a.delay);
        }
      }, a.onPointerMove = function (r) {
        for (var s = a.bbox, u = r.changedTouches || l.getCoords(r), h = u.length, p = 0; p < h; p++) {
          var m = u[p], y = m.identifier || 1 / 0, x = a.tracker[y];
          if (x) {
            var S = m.pageX - s.x1, k = m.pageY - s.y1, D = S - x.start.x, M = k - x.start.y, C = Math.sqrt(D * D + M * M);
            if (!(S > 0 && S < s.width && k > 0 && k < s.height && C <= a.driftDeviance)) return F.remove(a.doc, "mousemove", a.onPointerMove), void (a.cancel = !0);
          }
        }
      }, a.onPointerUp = function (r) {
        if (l.pointerEnd(r, i, a)) {
          if (clearTimeout(f), F.remove(a.doc, "mousemove", a.onPointerMove), F.remove(a.doc, "mouseup", a.onPointerUp), r.cancelBubble && ++r.cancelBubbleCount > 1) return;
          if (a.gesture === "longpress") return void (i.state === "start" && (i.state = "end", a.listener(r, i)));
          if (a.cancel || (/* @__PURE__ */ new Date()).getTime() - c > a.timeout) return;
          var s = a.gestureFingers;
          a.minFingers <= s && a.maxFingers >= s && (i.state = "tap", i.fingers = a.gestureFingers, a.listener(r, i));
        }
      };
      var i = l.pointerSetup(a);
      return F.add(a.target, "mousedown", a.onPointerDown), i;
    }, F.Gesture = F.Gesture || {}, F.Gesture._gestureHandlers = F.Gesture._gestureHandlers || {}, F.Gesture._gestureHandlers.tap = l.tap, F.Gesture._gestureHandlers.longpress = l.longpress, l;
  }(F.proxy), F === void 0 && (F = {}), F.proxy === void 0 && (F.proxy = {}), F.proxy = function (l) {
    return l.wheelPreventElasticBounce = function (a) {
      a && (typeof a == "string" && (a = document.querySelector(a)), F.add(a, "wheel", function (c, f) {
        f.preventElasticBounce(), F.stop(c);
      }));
    }, l.wheel = function (a) {
      var c, f = a.timeout || 150, i = 0, r = {
        gesture: "wheel", state: "start", wheelDelta: 0, target: a.target, listener: a.listener, preventElasticBounce: function (m) {
          var y = this.target, x = y.scrollTop;
          (x + y.offsetHeight === y.scrollHeight && this.wheelDelta <= 0 || x === 0 && this.wheelDelta >= 0) && F.cancel(m), F.stop(m);
        }, add: function () {
          a.target[u](p, s, !1);
        }, remove: function () {
          a.target[h](p, s, !1);
        }
      }, s = function (m) {
        m = m || window.event, r.state = i++ ? "change" : "start", r.wheelDelta = m.detail ? -20 * m.detail : m.wheelDelta, a.listener(m, r), clearTimeout(c), c = setTimeout(function () {
          i = 0, r.state = "end", r.wheelDelta = 0, a.listener(m, r);
        }, f);
      }, u = document.addEventListener ? "addEventListener" : "attachEvent", h = document.removeEventListener ? "removeEventListener" : "detachEvent", p = F.getEventSupport("mousewheel") ? "mousewheel" : "DOMMouseScroll";
      return a.target[u](p, s, !1), r;
    }, F.Gesture = F.Gesture || {}, F.Gesture._gestureHandlers = F.Gesture._gestureHandlers || {}, F.Gesture._gestureHandlers.wheel = l.wheel, l;
  }(F.proxy), Et === void 0) var Et = {};
  function Ht(l, a) {
    var c = l.canvas, f = a.targetCanvas, i = f.getContext("2d");
    i.translate(0, f.height), i.scale(1, -1);
    var r = c.height - f.height;
    i.drawImage(c, 0, r, f.width, f.height, 0, 0, f.width, f.height);
  }
  function It(l, a) {
    var c = a.targetCanvas.getContext("2d"), f = a.destinationWidth, i = a.destinationHeight, r = f * i * 4, s = new Uint8Array(this.imageBuffer, 0, r), u = new Uint8ClampedArray(this.imageBuffer, 0, r);
    l.readPixels(0, 0, f, i, l.RGBA, l.UNSIGNED_BYTE, s);
    var h = new ImageData(u, f, i);
    c.putImageData(h, 0, 0);
  }
  Et.proxy === void 0 && (Et.proxy = {}), Et.proxy = function (l) {
    return l.orientation = function (a) {
      var c = {
        gesture: "orientationchange", previous: null, current: window.orientation, target: a.target, listener: a.listener, remove: function () {
          window.removeEventListener("orientationchange", f, !1);
        }
      }, f = function (i) {
        c.previous = c.current, c.current = window.orientation, c.previous === null || c.previous == c.current || a.listener(i, c);
      };
      return window.DeviceOrientationEvent && window.addEventListener("orientationchange", f, !1), c;
    }, Et.Gesture = Et.Gesture || {}, Et.Gesture._gestureHandlers = Et.Gesture._gestureHandlers || {}, Et.Gesture._gestureHandlers.orientation = l.orientation, l;
  }(Et.proxy), function () {
    function l(c, f) {
      if (this.__eventListeners[c]) {
        var i = this.__eventListeners[c];
        f ? i[i.indexOf(f)] = !1 : v.util.array.fill(i, !1);
      }
    }
    function a(c, f) {
      var i = (function () {
        f.apply(this, arguments), this.off(c, i);
      }).bind(this);
      this.on(c, i);
    }
    v.Observable = {
      fire: function (c, f) {
        if (!this.__eventListeners) return this;
        var i = this.__eventListeners[c];
        if (!i) return this;
        for (var r = 0, s = i.length; r < s; r++) i[r] && i[r].call(this, f || {});
        return this.__eventListeners[c] = i.filter(function (u) {
          return u !== !1;
        }), this;
      }, on: function (c, f) {
        if (this.__eventListeners || (this.__eventListeners = {}), arguments.length === 1) for (var i in c) this.on(i, c[i]);
        else this.__eventListeners[c] || (this.__eventListeners[c] = []), this.__eventListeners[c].push(f);
        return this;
      }, once: function (c, f) {
        if (arguments.length === 1) for (var i in c) a.call(this, i, c[i]);
        else a.call(this, c, f);
        return this;
      }, off: function (c, f) {
        if (!this.__eventListeners) return this;
        if (arguments.length === 0) for (c in this.__eventListeners) l.call(this, c);
        else if (arguments.length === 1 && typeof arguments[0] == "object") for (var i in c) l.call(this, i, c[i]);
        else l.call(this, c, f);
        return this;
      }
    };
  }(), v.Collection = {
    _objects: [], add: function () {
      if (this._objects.push.apply(this._objects, arguments), this._onObjectAdded) for (var l = 0, a = arguments.length; l < a; l++) this._onObjectAdded(arguments[l]);
      return this.renderOnAddRemove && this.requestRenderAll(), this;
    }, insertAt: function (l, a, c) {
      var f = this._objects;
      return c ? f[a] = l : f.splice(a, 0, l), this._onObjectAdded && this._onObjectAdded(l), this.renderOnAddRemove && this.requestRenderAll(), this;
    }, remove: function () {
      for (var l, a = this._objects, c = !1, f = 0, i = arguments.length; f < i; f++) (l = a.indexOf(arguments[f])) !== -1 && (c = !0, a.splice(l, 1), this._onObjectRemoved && this._onObjectRemoved(arguments[f]));
      return this.renderOnAddRemove && c && this.requestRenderAll(), this;
    }, forEachObject: function (l, a) {
      for (var c = this.getObjects(), f = 0, i = c.length; f < i; f++) l.call(a, c[f], f, c);
      return this;
    }, getObjects: function (l) {
      return l === void 0 ? this._objects.concat() : this._objects.filter(function (a) {
        return a.type === l;
      });
    }, item: function (l) {
      return this._objects[l];
    }, isEmpty: function () {
      return this._objects.length === 0;
    }, size: function () {
      return this._objects.length;
    }, contains: function (l, a) {
      return this._objects.indexOf(l) > -1 || !!a && this._objects.some(function (c) {
        return typeof c.contains == "function" && c.contains(l, !0);
      });
    }, complexity: function () {
      return this._objects.reduce(function (l, a) {
        return l += a.complexity ? a.complexity() : 0;
      }, 0);
    }
  }, v.CommonMethods = {
    _setOptions: function (l) {
      for (var a in l) this.set(a, l[a]);
    }, _initGradient: function (l, a) {
      !l || !l.colorStops || l instanceof v.Gradient || this.set(a, new v.Gradient(l));
    }, _initPattern: function (l, a, c) {
      !l || !l.source || l instanceof v.Pattern ? c && c() : this.set(a, new v.Pattern(l, c));
    }, _setObject: function (l) {
      for (var a in l) this._set(a, l[a]);
    }, set: function (l, a) {
      return typeof l == "object" ? this._setObject(l) : this._set(l, a), this;
    }, _set: function (l, a) {
      this[l] = a;
    }, toggle: function (l) {
      var a = this.get(l);
      return typeof a == "boolean" && this.set(l, !a), this;
    }, get: function (l) {
      return this[l];
    }
  }, d = t, g = Math.sqrt, b = Math.atan2, _ = Math.pow, w = Math.PI / 180, E = Math.PI / 2, v.util = {
    cos: function (l) {
      if (l === 0) return 1;
      switch (l < 0 && (l = -l), l / E) {
        case 1:
        case 3:
          return 0;
        case 2:
          return -1;
      }
      return Math.cos(l);
    }, sin: function (l) {
      if (l === 0) return 0;
      var a = 1;
      switch (l < 0 && (a = -1), l / E) {
        case 1:
          return a;
        case 2:
          return 0;
        case 3:
          return -a;
      }
      return Math.sin(l);
    }, removeFromArray: function (l, a) {
      var c = l.indexOf(a);
      return c !== -1 && l.splice(c, 1), l;
    }, getRandomInt: function (l, a) {
      return Math.floor(Math.random() * (a - l + 1)) + l;
    }, degreesToRadians: function (l) {
      return l * w;
    }, radiansToDegrees: function (l) {
      return l / w;
    }, rotatePoint: function (l, a, c) {
      var f = new v.Point(l.x - a.x, l.y - a.y), i = v.util.rotateVector(f, c);
      return new v.Point(i.x, i.y).addEquals(a);
    }, rotateVector: function (l, a) {
      var c = v.util.sin(a), f = v.util.cos(a);
      return { x: l.x * f - l.y * c, y: l.x * c + l.y * f };
    }, transformPoint: function (l, a, c) {
      return c ? new v.Point(a[0] * l.x + a[2] * l.y, a[1] * l.x + a[3] * l.y) : new v.Point(a[0] * l.x + a[2] * l.y + a[4], a[1] * l.x + a[3] * l.y + a[5]);
    }, makeBoundingBoxFromPoints: function (l, a) {
      if (a) for (var c = 0; c < l.length; c++) l[c] = v.util.transformPoint(l[c], a);
      var f = [l[0].x, l[1].x, l[2].x, l[3].x], i = v.util.array.min(f), r = v.util.array.max(f) - i, s = [l[0].y, l[1].y, l[2].y, l[3].y], u = v.util.array.min(s);
      return { left: i, top: u, width: r, height: v.util.array.max(s) - u };
    }, invertTransform: function (l) {
      var a = 1 / (l[0] * l[3] - l[1] * l[2]), c = [a * l[3], -a * l[1], -a * l[2], a * l[0]], f = v.util.transformPoint({ x: l[4], y: l[5] }, c, !0);
      return c[4] = -f.x, c[5] = -f.y, c;
    }, toFixed: function (l, a) {
      return parseFloat(Number(l).toFixed(a));
    }, parseUnit: function (l, a) {
      var c = /\D{0,2}$/.exec(l), f = parseFloat(l);
      switch (a || (a = v.Text.DEFAULT_SVG_FONT_SIZE), c[0]) {
        case "mm":
          return f * v.DPI / 25.4;
        case "cm":
          return f * v.DPI / 2.54;
        case "in":
          return f * v.DPI;
        case "pt":
          return f * v.DPI / 72;
        case "pc":
          return f * v.DPI / 72 * 12;
        case "em":
          return f * a;
        default:
          return f;
      }
    }, falseFunction: function () {
      return !1;
    }, getKlass: function (l, a) {
      return l = v.util.string.camelize(l.charAt(0).toUpperCase() + l.slice(1)), v.util.resolveNamespace(a)[l];
    }, getSvgAttributes: function (l) {
      var a = ["instantiated_by_use", "style", "id", "class"];
      switch (l) {
        case "linearGradient":
          a = a.concat(["x1", "y1", "x2", "y2", "gradientUnits", "gradientTransform"]);
          break;
        case "radialGradient":
          a = a.concat(["gradientUnits", "gradientTransform", "cx", "cy", "r", "fx", "fy", "fr"]);
          break;
        case "stop":
          a = a.concat(["offset", "stop-color", "stop-opacity"]);
      }
      return a;
    }, resolveNamespace: function (l) {
      if (!l) return v;
      var a, c = l.split("."), f = c.length, i = d || v.window;
      for (a = 0; a < f; ++a) i = i[c[a]];
      return i;
    }, loadImage: function (l, a, c, f) {
      if (l) {
        var i = v.util.createImage(), r = function () {
          a && a.call(c, i, !1), i = i.onload = i.onerror = null;
        };
        i.onload = r, i.onerror = function () {
          v.log("Error loading " + i.src), a && a.call(c, null, !0), i = i.onload = i.onerror = null;
        }, l.indexOf("data") !== 0 && f != null && (i.crossOrigin = f), l.substring(0, 14) === "data:image/svg" && (i.onload = null, v.util.loadImageInDom(i, r)), i.src = l;
      } else a && a.call(c, l);
    }, loadImageInDom: function (l, a) {
      var c = v.document.createElement("div");
      c.style.width = c.style.height = "1px", c.style.left = c.style.top = "-100%", c.style.position = "absolute", c.appendChild(l), v.document.querySelector("body").appendChild(c), l.onload = function () {
        a(), c.parentNode.removeChild(c), c = null;
      };
    }, enlivenObjects: function (l, a, c, f) {
      var i = [], r = 0, s = (l = l || []).length;
      function u() {
        ++r === s && a && a(i.filter(function (h) {
          return h;
        }));
      }
      s ? l.forEach(function (h, p) {
        h && h.type ? v.util.getKlass(h.type, c).fromObject(h, function (m, y) {
          y || (i[p] = m), f && f(h, m, y), u();
        }) : u();
      }) : a && a(i);
    }, enlivenPatterns: function (l, a) {
      function c() {
        ++i === r && a && a(f);
      }
      var f = [], i = 0, r = (l = l || []).length;
      r ? l.forEach(function (s, u) {
        s && s.source ? new v.Pattern(s, function (h) {
          f[u] = h, c();
        }) : (f[u] = s, c());
      }) : a && a(f);
    }, groupSVGElements: function (l, a, c) {
      var f;
      return l && l.length === 1 ? l[0] : (a && (a.width && a.height ? a.centerPoint = { x: a.width / 2, y: a.height / 2 } : (delete a.width, delete a.height)), f = new v.Group(l, a), c !== void 0 && (f.sourcePath = c), f);
    }, populateWithProperties: function (l, a, c) {
      if (c && Object.prototype.toString.call(c) === "[object Array]") for (var f = 0, i = c.length; f < i; f++) c[f] in l && (a[c[f]] = l[c[f]]);
    }, drawDashedLine: function (l, a, c, f, i, r) {
      var s = f - a, u = i - c, h = g(s * s + u * u), p = b(u, s), m = r.length, y = 0, x = !0;
      for (l.save(), l.translate(a, c), l.moveTo(0, 0), l.rotate(p), a = 0; h > a;) (a += r[y++ % m]) > h && (a = h), l[x ? "lineTo" : "moveTo"](a, 0), x = !x;
      l.restore();
    }, createCanvasElement: function () {
      return v.document.createElement("canvas");
    }, copyCanvasElement: function (l) {
      var a = v.util.createCanvasElement();
      return a.width = l.width, a.height = l.height, a.getContext("2d").drawImage(l, 0, 0), a;
    }, toDataURL: function (l, a, c) {
      return l.toDataURL("image/" + a, c);
    }, createImage: function () {
      return v.document.createElement("img");
    }, multiplyTransformMatrices: function (l, a, c) {
      return [l[0] * a[0] + l[2] * a[1], l[1] * a[0] + l[3] * a[1], l[0] * a[2] + l[2] * a[3], l[1] * a[2] + l[3] * a[3], c ? 0 : l[0] * a[4] + l[2] * a[5] + l[4], c ? 0 : l[1] * a[4] + l[3] * a[5] + l[5]];
    }, qrDecompose: function (l) {
      var a = b(l[1], l[0]), c = _(l[0], 2) + _(l[1], 2), f = g(c), i = (l[0] * l[3] - l[2] * l[1]) / f, r = b(l[0] * l[2] + l[1] * l[3], c);
      return { angle: a / w, scaleX: f, scaleY: i, skewX: r / w, skewY: 0, translateX: l[4], translateY: l[5] };
    }, calcRotateMatrix: function (l) {
      if (!l.angle) return v.iMatrix.concat();
      var a = v.util.degreesToRadians(l.angle), c = v.util.cos(a), f = v.util.sin(a);
      return [c, f, -f, c, 0, 0];
    }, calcDimensionsMatrix: function (l) {
      var a = l.scaleX === void 0 ? 1 : l.scaleX, c = l.scaleY === void 0 ? 1 : l.scaleY, f = [l.flipX ? -a : a, 0, 0, l.flipY ? -c : c, 0, 0], i = v.util.multiplyTransformMatrices, r = v.util.degreesToRadians;
      return l.skewX && (f = i(f, [1, 0, Math.tan(r(l.skewX)), 1], !0)), l.skewY && (f = i(f, [1, Math.tan(r(l.skewY)), 0, 1], !0)), f;
    }, composeMatrix: function (l) {
      var a = [1, 0, 0, 1, l.translateX || 0, l.translateY || 0], c = v.util.multiplyTransformMatrices;
      return l.angle && (a = c(a, v.util.calcRotateMatrix(l))), (l.scaleX !== 1 || l.scaleY !== 1 || l.skewX || l.skewY || l.flipX || l.flipY) && (a = c(a, v.util.calcDimensionsMatrix(l))), a;
    }, resetObjectTransform: function (l) {
      l.scaleX = 1, l.scaleY = 1, l.skewX = 0, l.skewY = 0, l.flipX = !1, l.flipY = !1, l.rotate(0);
    }, saveObjectTransform: function (l) {
      return { scaleX: l.scaleX, scaleY: l.scaleY, skewX: l.skewX, skewY: l.skewY, angle: l.angle, left: l.left, flipX: l.flipX, flipY: l.flipY, top: l.top };
    }, isTransparent: function (l, a, c, f) {
      f > 0 && (a > f ? a -= f : a = 0, c > f ? c -= f : c = 0);
      var i, r = !0, s = l.getImageData(a, c, 2 * f || 1, 2 * f || 1), u = s.data.length;
      for (i = 3; i < u && (r = s.data[i] <= 0); i += 4);
      return s = null, r;
    }, parsePreserveAspectRatioAttribute: function (l) {
      var a, c = "meet", f = l.split(" ");
      return f && f.length && ((c = f.pop()) !== "meet" && c !== "slice" ? (a = c, c = "meet") : f.length && (a = f.pop())), { meetOrSlice: c, alignX: a !== "none" ? a.slice(1, 4) : "none", alignY: a !== "none" ? a.slice(5, 8) : "none" };
    }, clearFabricFontCache: function (l) {
      (l = (l || "").toLowerCase()) ? v.charWidthsCache[l] && delete v.charWidthsCache[l] : v.charWidthsCache = {};
    }, limitDimsByArea: function (l, a) {
      var c = Math.sqrt(a * l), f = Math.floor(a / c);
      return { x: Math.floor(c), y: f };
    }, capValue: function (l, a, c) {
      return Math.max(l, Math.min(a, c));
    }, findScaleToFit: function (l, a) {
      return Math.min(a.width / l.width, a.height / l.height);
    }, findScaleToCover: function (l, a) {
      return Math.max(a.width / l.width, a.height / l.height);
    }, matrixToSVG: function (l) {
      return "matrix(" + l.map(function (a) {
        return v.util.toFixed(a, v.Object.NUM_FRACTION_DIGITS);
      }).join(" ") + ")";
    }, removeTransformFromObject: function (l, a) {
      var c = v.util.invertTransform(a), f = v.util.multiplyTransformMatrices(c, l.calcOwnMatrix());
      v.util.applyTransformToObject(l, f);
    }, addTransformToObject: function (l, a) {
      v.util.applyTransformToObject(l, v.util.multiplyTransformMatrices(a, l.calcOwnMatrix()));
    }, applyTransformToObject: function (l, a) {
      var c = v.util.qrDecompose(a), f = new v.Point(c.translateX, c.translateY);
      l.flipX = !1, l.flipY = !1, l.set("scaleX", c.scaleX), l.set("scaleY", c.scaleY), l.skewX = c.skewX, l.skewY = c.skewY, l.angle = c.angle, l.setPositionByOrigin(f, "center", "center");
    }, sizeAfterTransform: function (l, a, c) {
      var f = l / 2, i = a / 2, r = [{ x: -f, y: -i }, { x: f, y: -i }, { x: -f, y: i }, { x: f, y: i }], s = v.util.calcDimensionsMatrix(c), u = v.util.makeBoundingBoxFromPoints(r, s);
      return { x: u.width, y: u.height };
    }
  }, v.util.createAccessors = function (l) {
    var a, c, f, i, r, s = l.prototype;
    for (a = s.stateProperties.length; a--;) i = "set" + (f = (c = s.stateProperties[a]).charAt(0).toUpperCase() + c.slice(1)), s[r = "get" + f] || (s[r] = new Function('return this.get("' + c + '")')), s[i] || (s[i] = function (u) {
      return new Function("value", 'return this.set("' + u + '", value)');
    }(c));
  }, function () {
    var l = Array.prototype.join, a = { m: 2, l: 2, h: 1, v: 1, c: 6, s: 4, q: 4, t: 2, a: 7 }, c = { m: "l", M: "L" };
    function f(M, C, O, N, W, X, I, T, R, Q, K) {
      var U = v.util.cos(M), B = v.util.sin(M), Z = v.util.cos(C), z = v.util.sin(C), V = O * W * Z - N * X * z + I, et = N * W * Z + O * X * z + T;
      return ["C", Q + R * (-O * W * B - N * X * U), K + R * (-N * W * B + O * X * U), V + R * (O * W * z + N * X * Z), et + R * (N * W * z - O * X * Z), V, et];
    }
    function i(M, C, O, N, W, X, I) {
      var T = Math.PI, R = I * T / 180, Q = v.util.sin(R), K = v.util.cos(R), U = 0, B = 0, Z = -K * M * 0.5 - Q * C * 0.5, z = -K * C * 0.5 + Q * M * 0.5, V = (O = Math.abs(O)) * O, et = (N = Math.abs(N)) * N, lt = z * z, at = Z * Z, ft = V * et - V * lt - et * at, ot = 0;
      if (ft < 0) {
        var ct = Math.sqrt(1 - ft / (V * et));
        O *= ct, N *= ct;
      } else ot = (W === X ? -1 : 1) * Math.sqrt(ft / (V * lt + et * at));
      var st = ot * O * z / N, yt = -ot * N * Z / O, gt = K * st - Q * yt + 0.5 * M, Tt = Q * st + K * yt + 0.5 * C, kt = r(1, 0, (Z - st) / O, (z - yt) / N), jt = r((Z - st) / O, (z - yt) / N, (-Z - st) / O, (-z - yt) / N);
      X === 0 && jt > 0 ? jt -= 2 * T : X === 1 && jt < 0 && (jt += 2 * T);
      for (var Kt = Math.ceil(Math.abs(jt / T * 2)), Rt = [], Bt = jt / Kt, se = 8 / 3 * Math.sin(Bt / 4) * Math.sin(Bt / 4) / Math.sin(Bt / 2), zt = kt + Bt, ie = 0; ie < Kt; ie++) Rt[ie] = f(kt, zt, K, Q, O, N, gt, Tt, se, U, B), U = Rt[ie][5], B = Rt[ie][6], kt = zt, zt += Bt;
      return Rt;
    }
    function r(M, C, O, N) {
      var W = Math.atan2(C, M), X = Math.atan2(N, O);
      return X >= W ? X - W : 2 * Math.PI - (W - X);
    }
    function s(M, C, O, N, W, X, I, T) {
      var R;
      if (v.cachesBoundsOfCurve && (R = l.call(arguments), v.boundsOfCurveCache[R])) return v.boundsOfCurveCache[R];
      var Q, K, U, B, Z, z, V, et, lt = Math.sqrt, at = Math.min, ft = Math.max, ot = Math.abs, ct = [], st = [[], []];
      K = 6 * M - 12 * O + 6 * W, Q = -3 * M + 9 * O - 9 * W + 3 * I, U = 3 * O - 3 * M;
      for (var yt = 0; yt < 2; ++yt) if (yt > 0 && (K = 6 * C - 12 * N + 6 * X, Q = -3 * C + 9 * N - 9 * X + 3 * T, U = 3 * N - 3 * C), ot(Q) < 1e-12) {
        if (ot(K) < 1e-12) continue;
        0 < (B = -U / K) && B < 1 && ct.push(B);
      } else (V = K * K - 4 * U * Q) < 0 || (0 < (Z = (-K + (et = lt(V))) / (2 * Q)) && Z < 1 && ct.push(Z), 0 < (z = (-K - et) / (2 * Q)) && z < 1 && ct.push(z));
      for (var gt, Tt, kt, jt = ct.length, Kt = jt; jt--;) gt = (kt = 1 - (B = ct[jt])) * kt * kt * M + 3 * kt * kt * B * O + 3 * kt * B * B * W + B * B * B * I, st[0][jt] = gt, Tt = kt * kt * kt * C + 3 * kt * kt * B * N + 3 * kt * B * B * X + B * B * B * T, st[1][jt] = Tt;
      st[0][Kt] = M, st[1][Kt] = C, st[0][Kt + 1] = I, st[1][Kt + 1] = T;
      var Rt = [{ x: at.apply(null, st[0]), y: at.apply(null, st[1]) }, { x: ft.apply(null, st[0]), y: ft.apply(null, st[1]) }];
      return v.cachesBoundsOfCurve && (v.boundsOfCurveCache[R] = Rt), Rt;
    }
    function u(M, C, O) {
      for (var N = O[1], W = O[2], X = O[3], I = O[4], T = O[5], R = i(O[6] - M, O[7] - C, N, W, I, T, X), Q = 0, K = R.length; Q < K; Q++) R[Q][1] += M, R[Q][2] += C, R[Q][3] += M, R[Q][4] += C, R[Q][5] += M, R[Q][6] += C;
      return R;
    }
    function h(M, C, O, N) {
      return Math.sqrt((O - M) * (O - M) + (N - C) * (N - C));
    }
    function p(M, C, O, N, W, X, I, T) {
      return function (R) {
        var Q, K = (Q = R) * Q * Q, U = function (z) {
          return 3 * z * z * (1 - z);
        }(R), B = function (z) {
          return 3 * z * (1 - z) * (1 - z);
        }(R), Z = function (z) {
          return (1 - z) * (1 - z) * (1 - z);
        }(R);
        return { x: I * K + W * U + O * B + M * Z, y: T * K + X * U + N * B + C * Z };
      };
    }
    function m(M, C, O, N, W, X, I, T) {
      return function (R) {
        var Q = 1 - R, K = 3 * Q * Q * (O - M) + 6 * Q * R * (W - O) + 3 * R * R * (I - W), U = 3 * Q * Q * (N - C) + 6 * Q * R * (X - N) + 3 * R * R * (T - X);
        return Math.atan2(U, K);
      };
    }
    function y(M, C, O, N, W, X) {
      return function (I) {
        var T, R = (T = I) * T, Q = function (U) {
          return 2 * U * (1 - U);
        }(I), K = function (U) {
          return (1 - U) * (1 - U);
        }(I);
        return { x: W * R + O * Q + M * K, y: X * R + N * Q + C * K };
      };
    }
    function x(M, C, O, N, W, X) {
      return function (I) {
        var T = 1 - I, R = 2 * T * (O - M) + 2 * I * (W - O), Q = 2 * T * (N - C) + 2 * I * (X - N);
        return Math.atan2(Q, R);
      };
    }
    function S(M, C, O) {
      var N, W, X = { x: C, y: O }, I = 0;
      for (W = 1; W <= 100; W += 1) N = M(W / 100), I += h(X.x, X.y, N.x, N.y), X = N;
      return I;
    }
    function k(M, C) {
      for (var O, N, W, X = 0, I = 0, T = M.iterator, R = { x: M.x, y: M.y }, Q = 0.01, K = M.angleFinder; I < C && X <= 1 && Q > 1e-4;) O = T(X), W = X, (N = h(R.x, R.y, O.x, O.y)) + I > C ? X -= Q /= 2 : (R = O, X += Q, I += N);
      return O.angle = K(W), O;
    }
    function D(M) {
      for (var C, O, N, W, X = 0, I = M.length, T = 0, R = 0, Q = 0, K = 0, U = [], B = 0; B < I; B++) {
        switch (N = { x: T, y: R, command: (C = M[B])[0] }, C[0]) {
          case "M":
            N.length = 0, Q = T = C[1], K = R = C[2];
            break;
          case "L":
            N.length = h(T, R, C[1], C[2]), T = C[1], R = C[2];
            break;
          case "C":
            O = p(T, R, C[1], C[2], C[3], C[4], C[5], C[6]), W = m(T, R, C[1], C[2], C[3], C[4], C[5], C[6]), N.iterator = O, N.angleFinder = W, N.length = S(O, T, R), T = C[5], R = C[6];
            break;
          case "Q":
            O = y(T, R, C[1], C[2], C[3], C[4]), W = x(T, R, C[1], C[2], C[3], C[4]), N.iterator = O, N.angleFinder = W, N.length = S(O, T, R), T = C[3], R = C[4];
            break;
          case "Z":
          case "z":
            N.destX = Q, N.destY = K, N.length = h(T, R, Q, K), T = Q, R = K;
        }
        X += N.length, U.push(N);
      }
      return U.push({ length: X, x: T, y: R }), U;
    }
    v.util.joinPath = function (M) {
      return M.map(function (C) {
        return C.join(" ");
      }).join(" ");
    }, v.util.parsePath = function (M) {
      var C, O, N, W, X, I = [], T = [], R = v.rePathCommand, Q = "[-+]?(?:\\d*\\.\\d+|\\d+\\.?)(?:[eE][-+]?\\d+)?\\s*", K = "(" + Q + ")" + v.commaWsp, U = "([01])" + v.commaWsp + "?", B = new RegExp(K + "?" + K + "?" + K + U + U + K + "?(" + Q + ")", "g");
      if (!M || !M.match) return I;
      for (var Z, z = 0, V = (X = M.match(/[mzlhvcsqta][^mzlhvcsqta]*/gi)).length; z < V; z++) {
        W = (C = X[z]).slice(1).trim(), T.length = 0;
        var et = C.charAt(0);
        if (Z = [et], et.toLowerCase() === "a") for (var lt; lt = B.exec(W);) for (var at = 1; at < lt.length; at++) T.push(lt[at]);
        else for (; N = R.exec(W);) T.push(N[0]);
        at = 0;
        for (var ft = T.length; at < ft; at++) O = parseFloat(T[at]), isNaN(O) || Z.push(O);
        var ot = a[et.toLowerCase()], ct = c[et] || et;
        if (Z.length - 1 > ot) for (var st = 1, yt = Z.length; st < yt; st += ot) I.push([et].concat(Z.slice(st, st + ot))), et = ct;
        else I.push(Z);
      }
      return I;
    }, v.util.makePathSimpler = function (M) {
      var C, O, N, W, X, I, T = 0, R = 0, Q = M.length, K = 0, U = 0, B = [];
      for (O = 0; O < Q; ++O) {
        switch (N = !1, (C = M[O].slice(0))[0]) {
          case "l":
            C[0] = "L", C[1] += T, C[2] += R;
          case "L":
            T = C[1], R = C[2];
            break;
          case "h":
            C[1] += T;
          case "H":
            C[0] = "L", C[2] = R, T = C[1];
            break;
          case "v":
            C[1] += R;
          case "V":
            C[0] = "L", R = C[1], C[1] = T, C[2] = R;
            break;
          case "m":
            C[0] = "M", C[1] += T, C[2] += R;
          case "M":
            T = C[1], R = C[2], K = C[1], U = C[2];
            break;
          case "c":
            C[0] = "C", C[1] += T, C[2] += R, C[3] += T, C[4] += R, C[5] += T, C[6] += R;
          case "C":
            X = C[3], I = C[4], T = C[5], R = C[6];
            break;
          case "s":
            C[0] = "S", C[1] += T, C[2] += R, C[3] += T, C[4] += R;
          case "S":
            W === "C" ? (X = 2 * T - X, I = 2 * R - I) : (X = T, I = R), T = C[3], R = C[4], C[0] = "C", C[5] = C[3], C[6] = C[4], C[3] = C[1], C[4] = C[2], C[1] = X, C[2] = I, X = C[3], I = C[4];
            break;
          case "q":
            C[0] = "Q", C[1] += T, C[2] += R, C[3] += T, C[4] += R;
          case "Q":
            X = C[1], I = C[2], T = C[3], R = C[4];
            break;
          case "t":
            C[0] = "T", C[1] += T, C[2] += R;
          case "T":
            W === "Q" ? (X = 2 * T - X, I = 2 * R - I) : (X = T, I = R), C[0] = "Q", T = C[1], R = C[2], C[1] = X, C[2] = I, C[3] = T, C[4] = R;
            break;
          case "a":
            C[0] = "A", C[6] += T, C[7] += R;
          case "A":
            N = !0, B = B.concat(u(T, R, C)), T = C[6], R = C[7];
            break;
          case "z":
          case "Z":
            T = K, R = U;
        }
        N || B.push(C), W = C[0];
      }
      return B;
    }, v.util.getSmoothPathFromPoints = function (M, C) {
      var O, N = [], W = new v.Point(M[0].x, M[0].y), X = new v.Point(M[1].x, M[1].y), I = M.length, T = 1, R = 0, Q = I > 2;
      for (C = C || 0, Q && (T = M[2].x < X.x ? -1 : M[2].x === X.x ? 0 : 1, R = M[2].y < X.y ? -1 : M[2].y === X.y ? 0 : 1), N.push(["M", W.x - T * C, W.y - R * C]), O = 1; O < I; O++) {
        if (!W.eq(X)) {
          var K = W.midPointFrom(X);
          N.push(["Q", W.x, W.y, K.x, K.y]);
        }
        W = M[O], O + 1 < M.length && (X = M[O + 1]);
      }
      return Q && (T = W.x > M[O - 2].x ? 1 : W.x === M[O - 2].x ? 0 : -1, R = W.y > M[O - 2].y ? 1 : W.y === M[O - 2].y ? 0 : -1), N.push(["L", W.x + T * C, W.y + R * C]), N;
    }, v.util.getPathSegmentsInfo = D, v.util.getBoundsOfCurve = s, v.util.getPointOnPath = function (M, C, O) {
      O || (O = D(M));
      for (var N = 0; C - O[N].length > 0 && N < O.length - 2;) C -= O[N].length, N++;
      var W, X = O[N], I = C / X.length, T = X.command, R = M[N];
      switch (T) {
        case "M":
          return { x: X.x, y: X.y, angle: 0 };
        case "Z":
        case "z":
          return (W = new v.Point(X.x, X.y).lerp(new v.Point(X.destX, X.destY), I)).angle = Math.atan2(X.destY - X.y, X.destX - X.x), W;
        case "L":
          return (W = new v.Point(X.x, X.y).lerp(new v.Point(R[1], R[2]), I)).angle = Math.atan2(R[2] - X.y, R[1] - X.x), W;
        case "C":
        case "Q":
          return k(X, C);
      }
    }, v.util.transformPath = function (M, C, O) {
      return O && (C = v.util.multiplyTransformMatrices(C, [1, 0, 0, 1, -O.x, -O.y])), M.map(function (N) {
        for (var W = N.slice(0), X = {}, I = 1; I < N.length - 1; I += 2) X.x = N[I], X.y = N[I + 1], X = v.util.transformPoint(X, C), W[I] = X.x, W[I + 1] = X.y;
        return W;
      });
    }, v.util.fromArcToBeizers = u, v.util.getBoundsOfArc = function (M, C, O, N, W, X, I, T, R) {
      for (var Q, K = 0, U = 0, B = [], Z = i(T - M, R - C, O, N, X, I, W), z = 0, V = Z.length; z < V; z++) Q = s(K, U, Z[z][1], Z[z][2], Z[z][3], Z[z][4], Z[z][5], Z[z][6]), B.push({ x: Q[0].x + M, y: Q[0].y + C }), B.push({ x: Q[1].x + M, y: Q[1].y + C }), K = Z[z][5], U = Z[z][6];
      return B;
    }, v.util.drawArc = function (M, C, O, N) {
      u(C, O, N = N.slice(0).unshift("X")).forEach(function (W) {
        M.bezierCurveTo.apply(M, W.slice(1));
      });
    };
  }(), function () {
    var l = Array.prototype.slice;
    function a(c, f, i) {
      if (c && c.length !== 0) {
        var r = c.length - 1, s = f ? c[r][f] : c[r];
        if (f) for (; r--;) i(c[r][f], s) && (s = c[r][f]);
        else for (; r--;) i(c[r], s) && (s = c[r]);
        return s;
      }
    }
    v.util.array = {
      fill: function (c, f) {
        for (var i = c.length; i--;) c[i] = f;
        return c;
      }, invoke: function (c, f) {
        for (var i = l.call(arguments, 2), r = [], s = 0, u = c.length; s < u; s++) r[s] = i.length ? c[s][f].apply(c[s], i) : c[s][f].call(c[s]);
        return r;
      }, min: function (c, f) {
        return a(c, f, function (i, r) {
          return i < r;
        });
      }, max: function (c, f) {
        return a(c, f, function (i, r) {
          return i >= r;
        });
      }
    };
  }(), function () {
    function l(a, c, f) {
      if (f) if (!v.isLikelyNode && c instanceof Element) a = c;
      else if (c instanceof Array) {
        a = [];
        for (var i = 0, r = c.length; i < r; i++) a[i] = l({}, c[i], f);
      } else if (c && typeof c == "object") for (var s in c) s === "canvas" || s === "group" ? a[s] = null : c.hasOwnProperty(s) && (a[s] = l({}, c[s], f));
      else a = c;
      else for (var s in c) a[s] = c[s];
      return a;
    }
    v.util.object = {
      extend: l, clone: function (a, c) {
        return l({}, a, c);
      }
    }, v.util.object.extend(v.util, v.Observable);
  }(), function () {
    function l(a, c) {
      var f = a.charCodeAt(c);
      if (isNaN(f)) return "";
      if (f < 55296 || f > 57343) return a.charAt(c);
      if (55296 <= f && f <= 56319) {
        if (a.length <= c + 1) throw "High surrogate without following low surrogate";
        var i = a.charCodeAt(c + 1);
        if (56320 > i || i > 57343) throw "High surrogate without following low surrogate";
        return a.charAt(c) + a.charAt(c + 1);
      }
      if (c === 0) throw "Low surrogate without preceding high surrogate";
      var r = a.charCodeAt(c - 1);
      if (55296 > r || r > 56319) throw "Low surrogate without preceding high surrogate";
      return !1;
    }
    v.util.string = {
      camelize: function (a) {
        return a.replace(/-+(.)?/g, function (c, f) {
          return f ? f.toUpperCase() : "";
        });
      }, capitalize: function (a, c) {
        return a.charAt(0).toUpperCase() + (c ? a.slice(1) : a.slice(1).toLowerCase());
      }, escapeXml: function (a) {
        return a.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&apos;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      }, graphemeSplit: function (a) {
        var c, f = 0, i = [];
        for (f = 0; f < a.length; f++) (c = l(a, f)) !== !1 && i.push(c);
        return i;
      }
    };
  }(), function () {
    var l = Array.prototype.slice, a = function () {
    }, c = function () {
      for (var s in { toString: 1 }) if (s === "toString") return !1;
      return !0;
    }(), f = function (s, u, h) {
      for (var p in u) p in s.prototype && typeof s.prototype[p] == "function" && (u[p] + "").indexOf("callSuper") > -1 ? s.prototype[p] = /* @__PURE__ */ function (m) {
        return function () {
          var y = this.constructor.superclass;
          this.constructor.superclass = h;
          var x = u[m].apply(this, arguments);
          if (this.constructor.superclass = y, m !== "initialize") return x;
        };
      }(p) : s.prototype[p] = u[p], c && (u.toString !== Object.prototype.toString && (s.prototype.toString = u.toString), u.valueOf !== Object.prototype.valueOf && (s.prototype.valueOf = u.valueOf));
    };
    function i() {
    }
    function r(s) {
      for (var u = null, h = this; h.constructor.superclass;) {
        var p = h.constructor.superclass.prototype[s];
        if (h[s] !== p) {
          u = p;
          break;
        }
        h = h.constructor.superclass.prototype;
      }
      return u ? arguments.length > 1 ? u.apply(this, l.call(arguments, 1)) : u.call(this) : console.log("tried to callSuper " + s + ", method not found in prototype chain", this);
    }
    v.util.createClass = function () {
      var s = null, u = l.call(arguments, 0);
      function h() {
        this.initialize.apply(this, arguments);
      }
      typeof u[0] == "function" && (s = u.shift()), h.superclass = s, h.subclasses = [], s && (i.prototype = s.prototype, h.prototype = new i(), s.subclasses.push(h));
      for (var p = 0, m = u.length; p < m; p++) f(h, u[p], s);
      return h.prototype.initialize || (h.prototype.initialize = a), h.prototype.constructor = h, h.prototype.callSuper = r, h;
    };
  }(), j = !!v.document.createElement("div").attachEvent, H = ["touchstart", "touchmove", "touchend"], v.util.addListener = function (l, a, c, f) {
    l && l.addEventListener(a, c, !j && f);
  }, v.util.removeListener = function (l, a, c, f) {
    l && l.removeEventListener(a, c, !j && f);
  }, v.util.getPointer = function (l) {
    var a = l.target, c = v.util.getScrollLeftTop(a), f = function (i) {
      var r = i.changedTouches;
      return r && r[0] ? r[0] : i;
    }(l);
    return { x: f.clientX + c.left, y: f.clientY + c.top };
  }, v.util.isTouchEvent = function (l) {
    return H.indexOf(l.type) > -1 || l.pointerType === "touch";
  }, G = v.document.createElement("div"), J = typeof G.style.opacity == "string", nt = typeof G.style.filter == "string", rt = /alpha\s*\(\s*opacity\s*=\s*([^\)]+)\)/, vt = function (l) {
    return l;
  }, J ? vt = function (l, a) {
    return l.style.opacity = a, l;
  } : nt && (vt = function (l, a) {
    var c = l.style;
    return l.currentStyle && !l.currentStyle.hasLayout && (c.zoom = 1), rt.test(c.filter) ? (a = a >= 0.9999 ? "" : "alpha(opacity=" + 100 * a + ")", c.filter = c.filter.replace(rt, a)) : c.filter += " alpha(opacity=" + 100 * a + ")", l;
  }), v.util.setStyle = function (l, a) {
    var c = l.style;
    if (!c) return l;
    if (typeof a == "string") return l.style.cssText += ";" + a, a.indexOf("opacity") > -1 ? vt(l, a.match(/opacity:\s*(\d?\.?\d*)/)[1]) : l;
    for (var f in a) f === "opacity" ? vt(l, a[f]) : c[f === "float" || f === "cssFloat" ? c.styleFloat === void 0 ? "cssFloat" : "styleFloat" : f] = a[f];
    return l;
  }, function () {
    var l = Array.prototype.slice, a, c, f, i, r = function (h) {
      return l.call(h, 0);
    };
    try {
      a = r(v.document.childNodes) instanceof Array;
    } catch {
    }
    function s(h, p) {
      var m = v.document.createElement(h);
      for (var y in p) y === "class" ? m.className = p[y] : y === "for" ? m.htmlFor = p[y] : m.setAttribute(y, p[y]);
      return m;
    }
    function u(h) {
      for (var p = 0, m = 0, y = v.document.documentElement, x = v.document.body || { scrollLeft: 0, scrollTop: 0 }; h && (h.parentNode || h.host) && ((h = h.parentNode || h.host) === v.document ? (p = x.scrollLeft || y.scrollLeft || 0, m = x.scrollTop || y.scrollTop || 0) : (p += h.scrollLeft || 0, m += h.scrollTop || 0), h.nodeType !== 1 || h.style.position !== "fixed"););
      return { left: p, top: m };
    }
    a || (r = function (h) {
      for (var p = new Array(h.length), m = h.length; m--;) p[m] = h[m];
      return p;
    }), c = v.document.defaultView && v.document.defaultView.getComputedStyle ? function (h, p) {
      var m = v.document.defaultView.getComputedStyle(h, null);
      return m ? m[p] : void 0;
    } : function (h, p) {
      var m = h.style[p];
      return !m && h.currentStyle && (m = h.currentStyle[p]), m;
    }, f = v.document.documentElement.style, i = "userSelect" in f ? "userSelect" : "MozUserSelect" in f ? "MozUserSelect" : "WebkitUserSelect" in f ? "WebkitUserSelect" : "KhtmlUserSelect" in f ? "KhtmlUserSelect" : "", v.util.makeElementUnselectable = function (h) {
      return h.onselectstart !== void 0 && (h.onselectstart = v.util.falseFunction), i ? h.style[i] = "none" : typeof h.unselectable == "string" && (h.unselectable = "on"), h;
    }, v.util.makeElementSelectable = function (h) {
      return h.onselectstart !== void 0 && (h.onselectstart = null), i ? h.style[i] = "" : typeof h.unselectable == "string" && (h.unselectable = ""), h;
    }, v.util.setImageSmoothing = function (h, p) {
      h.imageSmoothingEnabled = h.imageSmoothingEnabled || h.webkitImageSmoothingEnabled || h.mozImageSmoothingEnabled || h.msImageSmoothingEnabled || h.oImageSmoothingEnabled, h.imageSmoothingEnabled = p;
    }, v.util.getById = function (h) {
      return typeof h == "string" ? v.document.getElementById(h) : h;
    }, v.util.toArray = r, v.util.addClass = function (h, p) {
      h && (" " + h.className + " ").indexOf(" " + p + " ") === -1 && (h.className += (h.className ? " " : "") + p);
    }, v.util.makeElement = s, v.util.wrapElement = function (h, p, m) {
      return typeof p == "string" && (p = s(p, m)), h.parentNode && h.parentNode.replaceChild(p, h), p.appendChild(h), p;
    }, v.util.getScrollLeftTop = u, v.util.getElementOffset = function (h) {
      var p, m, y = h && h.ownerDocument, x = { left: 0, top: 0 }, S = { left: 0, top: 0 }, k = { borderLeftWidth: "left", borderTopWidth: "top", paddingLeft: "left", paddingTop: "top" };
      if (!y) return S;
      for (var D in k) S[k[D]] += parseInt(c(h, D), 10) || 0;
      return p = y.documentElement, h.getBoundingClientRect !== void 0 && (x = h.getBoundingClientRect()), m = u(h), { left: x.left + m.left - (p.clientLeft || 0) + S.left, top: x.top + m.top - (p.clientTop || 0) + S.top };
    }, v.util.getNodeCanvas = function (h) {
      var p = v.jsdomImplForWrapper(h);
      return p._canvas || p._image;
    }, v.util.cleanUpJsdomNode = function (h) {
      if (v.isLikelyNode) {
        var p = v.jsdomImplForWrapper(h);
        p && (p._image = null, p._canvas = null, p._currentSrc = null, p._attributes = null, p._classList = null);
      }
    };
  }(), function () {
    function l() {
    }
    v.util.request = function (a, c) {
      c || (c = {});
      var f = c.method ? c.method.toUpperCase() : "GET", i = c.onComplete || function () {
      }, r = new v.window.XMLHttpRequest(), s = c.body || c.parameters;
      return r.onreadystatechange = function () {
        r.readyState === 4 && (i(r), r.onreadystatechange = l);
      }, f === "GET" && (s = null, typeof c.parameters == "string" && (a = function (u, h) {
        return u + (/\?/.test(u) ? "&" : "?") + h;
      }(a, c.parameters))), r.open(f, a, !0), f !== "POST" && f !== "PUT" || r.setRequestHeader("Content-Type", "application/x-www-form-urlencoded"), r.send(s), r;
    };
  }(), v.log = console.log, v.warn = console.warn, function () {
    function l() {
      return !1;
    }
    function a(r, s, u, h) {
      return -u * Math.cos(r / h * (Math.PI / 2)) + u + s;
    }
    var c = v.window.requestAnimationFrame || v.window.webkitRequestAnimationFrame || v.window.mozRequestAnimationFrame || v.window.oRequestAnimationFrame || v.window.msRequestAnimationFrame || function (r) {
      return v.window.setTimeout(r, 1e3 / 60);
    }, f = v.window.cancelAnimationFrame || v.window.clearTimeout;
    function i() {
      return c.apply(v.window, arguments);
    }
    v.util.animate = function (r) {
      var s = !1;
      return i(function (u) {
        r || (r = {});
        var h, p = u || +/* @__PURE__ */ new Date(), m = r.duration || 500, y = p + m, x = r.onChange || l, S = r.abort || l, k = r.onComplete || l, D = r.easing || a, M = "startValue" in r ? r.startValue : 0, C = "endValue" in r ? r.endValue : 100, O = r.byValue || C - M;
        r.onStart && r.onStart(), function N(W) {
          var X = (h = W || +/* @__PURE__ */ new Date()) > y ? m : h - p, I = X / m, T = D(X, M, O, m), R = Math.abs((T - M) / O);
          if (!s) {
            if (!S(T, R, I)) return h > y ? (x(C, 1, 1), void k(C, 1, 1)) : (x(T, R, I), void i(N));
            k(C, 1, 1);
          }
        }(p);
      }), function () {
        s = !0;
      };
    }, v.util.requestAnimFrame = i, v.util.cancelAnimFrame = function () {
      return f.apply(v.window, arguments);
    };
  }(), function () {
    function l(a, c, f) {
      var i = "rgba(" + parseInt(a[0] + f * (c[0] - a[0]), 10) + "," + parseInt(a[1] + f * (c[1] - a[1]), 10) + "," + parseInt(a[2] + f * (c[2] - a[2]), 10);
      return i += "," + (a && c ? parseFloat(a[3] + f * (c[3] - a[3])) : 1), i += ")";
    }
    v.util.animateColor = function (a, c, f, i) {
      var r = new v.Color(a).getSource(), s = new v.Color(c).getSource(), u = i.onComplete, h = i.onChange;
      return i = i || {}, v.util.animate(v.util.object.extend(i, {
        duration: f || 500, startValue: r, endValue: s, byValue: s, easing: function (p, m, y, x) {
          return l(m, y, i.colorEasing ? i.colorEasing(p, x) : 1 - Math.cos(p / x * (Math.PI / 2)));
        }, onComplete: function (p, m, y) {
          if (u) return u(l(s, s, 0), m, y);
        }, onChange: function (p, m, y) {
          if (h) {
            if (Array.isArray(p)) return h(l(p, p, 0), m, y);
            h(p, m, y);
          }
        }
      }));
    };
  }(), function () {
    function l(i, r, s, u) {
      return i < Math.abs(r) ? (i = r, u = s / 4) : u = r === 0 && i === 0 ? s / (2 * Math.PI) * Math.asin(1) : s / (2 * Math.PI) * Math.asin(r / i), { a: i, c: r, p: s, s: u };
    }
    function a(i, r, s) {
      return i.a * Math.pow(2, 10 * (r -= 1)) * Math.sin((r * s - i.s) * (2 * Math.PI) / i.p);
    }
    function c(i, r, s, u) {
      return s - f(u - i, 0, s, u) + r;
    }
    function f(i, r, s, u) {
      return (i /= u) < 1 / 2.75 ? s * (7.5625 * i * i) + r : i < 2 / 2.75 ? s * (7.5625 * (i -= 1.5 / 2.75) * i + 0.75) + r : i < 2.5 / 2.75 ? s * (7.5625 * (i -= 2.25 / 2.75) * i + 0.9375) + r : s * (7.5625 * (i -= 2.625 / 2.75) * i + 0.984375) + r;
    }
    v.util.ease = {
      easeInQuad: function (i, r, s, u) {
        return s * (i /= u) * i + r;
      }, easeOutQuad: function (i, r, s, u) {
        return -s * (i /= u) * (i - 2) + r;
      }, easeInOutQuad: function (i, r, s, u) {
        return (i /= u / 2) < 1 ? s / 2 * i * i + r : -s / 2 * (--i * (i - 2) - 1) + r;
      }, easeInCubic: function (i, r, s, u) {
        return s * (i /= u) * i * i + r;
      }, easeOutCubic: function (i, r, s, u) {
        return s * ((i = i / u - 1) * i * i + 1) + r;
      }, easeInOutCubic: function (i, r, s, u) {
        return (i /= u / 2) < 1 ? s / 2 * i * i * i + r : s / 2 * ((i -= 2) * i * i + 2) + r;
      }, easeInQuart: function (i, r, s, u) {
        return s * (i /= u) * i * i * i + r;
      }, easeOutQuart: function (i, r, s, u) {
        return -s * ((i = i / u - 1) * i * i * i - 1) + r;
      }, easeInOutQuart: function (i, r, s, u) {
        return (i /= u / 2) < 1 ? s / 2 * i * i * i * i + r : -s / 2 * ((i -= 2) * i * i * i - 2) + r;
      }, easeInQuint: function (i, r, s, u) {
        return s * (i /= u) * i * i * i * i + r;
      }, easeOutQuint: function (i, r, s, u) {
        return s * ((i = i / u - 1) * i * i * i * i + 1) + r;
      }, easeInOutQuint: function (i, r, s, u) {
        return (i /= u / 2) < 1 ? s / 2 * i * i * i * i * i + r : s / 2 * ((i -= 2) * i * i * i * i + 2) + r;
      }, easeInSine: function (i, r, s, u) {
        return -s * Math.cos(i / u * (Math.PI / 2)) + s + r;
      }, easeOutSine: function (i, r, s, u) {
        return s * Math.sin(i / u * (Math.PI / 2)) + r;
      }, easeInOutSine: function (i, r, s, u) {
        return -s / 2 * (Math.cos(Math.PI * i / u) - 1) + r;
      }, easeInExpo: function (i, r, s, u) {
        return i === 0 ? r : s * Math.pow(2, 10 * (i / u - 1)) + r;
      }, easeOutExpo: function (i, r, s, u) {
        return i === u ? r + s : s * (1 - Math.pow(2, -10 * i / u)) + r;
      }, easeInOutExpo: function (i, r, s, u) {
        return i === 0 ? r : i === u ? r + s : (i /= u / 2) < 1 ? s / 2 * Math.pow(2, 10 * (i - 1)) + r : s / 2 * (2 - Math.pow(2, -10 * --i)) + r;
      }, easeInCirc: function (i, r, s, u) {
        return -s * (Math.sqrt(1 - (i /= u) * i) - 1) + r;
      }, easeOutCirc: function (i, r, s, u) {
        return s * Math.sqrt(1 - (i = i / u - 1) * i) + r;
      }, easeInOutCirc: function (i, r, s, u) {
        return (i /= u / 2) < 1 ? -s / 2 * (Math.sqrt(1 - i * i) - 1) + r : s / 2 * (Math.sqrt(1 - (i -= 2) * i) + 1) + r;
      }, easeInElastic: function (i, r, s, u) {
        var h = 0;
        return i === 0 ? r : (i /= u) === 1 ? r + s : (h || (h = 0.3 * u), -a(l(s, s, h, 1.70158), i, u) + r);
      }, easeOutElastic: function (i, r, s, u) {
        var h = 0;
        if (i === 0) return r;
        if ((i /= u) === 1) return r + s;
        h || (h = 0.3 * u);
        var p = l(s, s, h, 1.70158);
        return p.a * Math.pow(2, -10 * i) * Math.sin((i * u - p.s) * (2 * Math.PI) / p.p) + p.c + r;
      }, easeInOutElastic: function (i, r, s, u) {
        var h = 0;
        if (i === 0) return r;
        if ((i /= u / 2) === 2) return r + s;
        h || (h = u * (0.3 * 1.5));
        var p = l(s, s, h, 1.70158);
        return i < 1 ? -0.5 * a(p, i, u) + r : p.a * Math.pow(2, -10 * (i -= 1)) * Math.sin((i * u - p.s) * (2 * Math.PI) / p.p) * 0.5 + p.c + r;
      }, easeInBack: function (i, r, s, u, h) {
        return h === void 0 && (h = 1.70158), s * (i /= u) * i * ((h + 1) * i - h) + r;
      }, easeOutBack: function (i, r, s, u, h) {
        return h === void 0 && (h = 1.70158), s * ((i = i / u - 1) * i * ((h + 1) * i + h) + 1) + r;
      }, easeInOutBack: function (i, r, s, u, h) {
        return h === void 0 && (h = 1.70158), (i /= u / 2) < 1 ? s / 2 * (i * i * ((1 + (h *= 1.525)) * i - h)) + r : s / 2 * ((i -= 2) * i * ((1 + (h *= 1.525)) * i + h) + 2) + r;
      }, easeInBounce: c, easeOutBounce: f, easeInOutBounce: function (i, r, s, u) {
        return i < u / 2 ? 0.5 * c(2 * i, 0, s, u) + r : 0.5 * f(2 * i - u, 0, s, u) + 0.5 * s + r;
      }
    };
  }(), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.object.extend, f = a.util.object.clone, i = a.util.toFixed, r = a.util.parseUnit, s = a.util.multiplyTransformMatrices, u = { cx: "left", x: "left", r: "radius", cy: "top", y: "top", display: "visible", visibility: "visible", transform: "transformMatrix", "fill-opacity": "fillOpacity", "fill-rule": "fillRule", "font-family": "fontFamily", "font-size": "fontSize", "font-style": "fontStyle", "font-weight": "fontWeight", "letter-spacing": "charSpacing", "paint-order": "paintFirst", "stroke-dasharray": "strokeDashArray", "stroke-dashoffset": "strokeDashOffset", "stroke-linecap": "strokeLineCap", "stroke-linejoin": "strokeLineJoin", "stroke-miterlimit": "strokeMiterLimit", "stroke-opacity": "strokeOpacity", "stroke-width": "strokeWidth", "text-decoration": "textDecoration", "text-anchor": "textAnchor", opacity: "opacity", "clip-path": "clipPath", "clip-rule": "clipRule", "vector-effect": "strokeUniform", "image-rendering": "imageSmoothing" }, h = { stroke: "strokeOpacity", fill: "fillOpacity" }, p = "font-size", m = "clip-path";
    function y(I) {
      return I in u ? u[I] : I;
    }
    function x(I, T, R, Q) {
      var K, U = Object.prototype.toString.call(T) === "[object Array]";
      if (I !== "fill" && I !== "stroke" || T !== "none") {
        if (I === "strokeUniform") return T === "non-scaling-stroke";
        if (I === "strokeDashArray") T = T === "none" ? null : T.replace(/,/g, " ").split(/\s+/).map(parseFloat);
        else if (I === "transformMatrix") T = R && R.transformMatrix ? s(R.transformMatrix, a.parseTransformAttribute(T)) : a.parseTransformAttribute(T);
        else if (I === "visible") T = T !== "none" && T !== "hidden", R && R.visible === !1 && (T = !1);
        else if (I === "opacity") T = parseFloat(T), R && R.opacity !== void 0 && (T *= R.opacity);
        else if (I === "textAnchor") T = T === "start" ? "left" : T === "end" ? "right" : "center";
        else if (I === "charSpacing") K = r(T, Q) / Q * 1e3;
        else if (I === "paintFirst") {
          var B = T.indexOf("fill"), Z = T.indexOf("stroke");
          T = "fill", (B > -1 && Z > -1 && Z < B || B === -1 && Z > -1) && (T = "stroke");
        } else {
          if (I === "href" || I === "xlink:href" || I === "font") return T;
          if (I === "imageSmoothing") return T === "optimizeQuality";
          K = U ? T.map(r) : r(T, Q);
        }
      } else T = "";
      return !U && isNaN(K) ? T : K;
    }
    function S(I) {
      return new RegExp("^(" + I.join("|") + ")\\b", "i");
    }
    function k(I, T) {
      var R, Q, K, U, B = [];
      for (K = 0, U = T.length; K < U; K++) R = T[K], Q = I.getElementsByTagName(R), B = B.concat(Array.prototype.slice.call(Q));
      return B;
    }
    function D(I, T) {
      var R, Q = !0;
      return (R = M(I, T.pop())) && T.length && (Q = function (K, U) {
        for (var B, Z = !0; K.parentNode && K.parentNode.nodeType === 1 && U.length;) Z && (B = U.pop()), Z = M(K = K.parentNode, B);
        return U.length === 0;
      }(I, T)), R && Q && T.length === 0;
    }
    function M(I, T) {
      var R, Q, K = I.nodeName, U = I.getAttribute("class"), B = I.getAttribute("id");
      if (R = new RegExp("^" + K, "i"), T = T.replace(R, ""), B && T.length && (R = new RegExp("#" + B + "(?![a-zA-Z\\-]+)", "i"), T = T.replace(R, "")), U && T.length) for (Q = (U = U.split(" ")).length; Q--;) R = new RegExp("\\." + U[Q] + "(?![a-zA-Z\\-]+)", "i"), T = T.replace(R, "");
      return T.length === 0;
    }
    function C(I, T) {
      var R;
      if (I.getElementById && (R = I.getElementById(T)), R) return R;
      var Q, K, U, B = I.getElementsByTagName("*");
      for (K = 0, U = B.length; K < U; K++) if (T === (Q = B[K]).getAttribute("id")) return Q;
    }
    a.svgValidTagNamesRegEx = S(["path", "circle", "polygon", "polyline", "ellipse", "rect", "line", "image", "text"]), a.svgViewBoxElementsRegEx = S(["symbol", "image", "marker", "pattern", "view", "svg"]), a.svgInvalidAncestorsRegEx = S(["pattern", "defs", "symbol", "metadata", "clipPath", "mask", "desc"]), a.svgValidParentsRegEx = S(["symbol", "g", "a", "svg", "clipPath", "defs"]), a.cssRules = {}, a.gradientDefs = {}, a.clipPaths = {}, a.parseTransformAttribute = function () {
      function I(Z, z, V) {
        Z[V] = Math.tan(a.util.degreesToRadians(z[0]));
      }
      var T = a.iMatrix, R = a.reNum, Q = a.commaWsp, K = "(?:" + ("(?:(matrix)\\s*\\(\\s*(" + R + ")" + Q + "(" + R + ")" + Q + "(" + R + ")" + Q + "(" + R + ")" + Q + "(" + R + ")" + Q + "(" + R + ")\\s*\\))") + "|" + ("(?:(translate)\\s*\\(\\s*(" + R + ")(?:" + Q + "(" + R + "))?\\s*\\))") + "|" + ("(?:(scale)\\s*\\(\\s*(" + R + ")(?:" + Q + "(" + R + "))?\\s*\\))") + "|" + ("(?:(rotate)\\s*\\(\\s*(" + R + ")(?:" + Q + "(" + R + ")" + Q + "(" + R + "))?\\s*\\))") + "|" + ("(?:(skewX)\\s*\\(\\s*(" + R + ")\\s*\\))") + "|" + ("(?:(skewY)\\s*\\(\\s*(" + R + ")\\s*\\))") + ")", U = new RegExp("^\\s*(?:" + ("(?:" + K + "(?:" + Q + "*" + K + ")*)") + "?)\\s*$"), B = new RegExp(K, "g");
      return function (Z) {
        var z = T.concat(), V = [];
        if (!Z || Z && !U.test(Z)) return z;
        Z.replace(B, function (lt) {
          var at = new RegExp(K).exec(lt).filter(function (ct) {
            return !!ct;
          }), ft = at[1], ot = at.slice(2).map(parseFloat);
          switch (ft) {
            case "translate":
              (function (ct, st) {
                ct[4] = st[0], st.length === 2 && (ct[5] = st[1]);
              })(z, ot);
              break;
            case "rotate":
              ot[0] = a.util.degreesToRadians(ot[0]), function (ct, st) {
                var yt = a.util.cos(st[0]), gt = a.util.sin(st[0]), Tt = 0, kt = 0;
                st.length === 3 && (Tt = st[1], kt = st[2]), ct[0] = yt, ct[1] = gt, ct[2] = -gt, ct[3] = yt, ct[4] = Tt - (yt * Tt - gt * kt), ct[5] = kt - (gt * Tt + yt * kt);
              }(z, ot);
              break;
            case "scale":
              (function (ct, st) {
                var yt = st[0], gt = st.length === 2 ? st[1] : st[0];
                ct[0] = yt, ct[3] = gt;
              })(z, ot);
              break;
            case "skewX":
              I(z, ot, 2);
              break;
            case "skewY":
              I(z, ot, 1);
              break;
            case "matrix":
              z = ot;
          }
          V.push(z.concat()), z = T.concat();
        });
        for (var et = V[0]; V.length > 1;) V.shift(), et = a.util.multiplyTransformMatrices(et, V[0]);
        return et;
      };
    }();
    var O = new RegExp("^\\s*(" + a.reNum + "+)\\s*,?\\s*(" + a.reNum + "+)\\s*,?\\s*(" + a.reNum + "+)\\s*,?\\s*(" + a.reNum + "+)\\s*$");
    function N(I) {
      if (!a.svgViewBoxElementsRegEx.test(I.nodeName)) return {};
      var T, R, Q, K, U, B, Z = I.getAttribute("viewBox"), z = 1, V = 1, et = I.getAttribute("width"), lt = I.getAttribute("height"), at = I.getAttribute("x") || 0, ft = I.getAttribute("y") || 0, ot = I.getAttribute("preserveAspectRatio") || "", ct = !Z || !(Z = Z.match(O)), st = !et || !lt || et === "100%" || lt === "100%", yt = ct && st, gt = {}, Tt = "", kt = 0, jt = 0;
      if (gt.width = 0, gt.height = 0, gt.toBeParsed = yt, ct && (at || ft) && I.parentNode && I.parentNode.nodeName !== "#document" && (Tt = " translate(" + r(at) + " " + r(ft) + ") ", U = (I.getAttribute("transform") || "") + Tt, I.setAttribute("transform", U), I.removeAttribute("x"), I.removeAttribute("y")), yt) return gt;
      if (ct) return gt.width = r(et), gt.height = r(lt), gt;
      if (T = -parseFloat(Z[1]), R = -parseFloat(Z[2]), Q = parseFloat(Z[3]), K = parseFloat(Z[4]), gt.minX = T, gt.minY = R, gt.viewBoxWidth = Q, gt.viewBoxHeight = K, st ? (gt.width = Q, gt.height = K) : (gt.width = r(et), gt.height = r(lt), z = gt.width / Q, V = gt.height / K), (ot = a.util.parsePreserveAspectRatioAttribute(ot)).alignX !== "none" && (ot.meetOrSlice === "meet" && (V = z = z > V ? V : z), ot.meetOrSlice === "slice" && (V = z = z > V ? z : V), kt = gt.width - Q * z, jt = gt.height - K * z, ot.alignX === "Mid" && (kt /= 2), ot.alignY === "Mid" && (jt /= 2), ot.alignX === "Min" && (kt = 0), ot.alignY === "Min" && (jt = 0)), z === 1 && V === 1 && T === 0 && R === 0 && at === 0 && ft === 0) return gt;
      if ((at || ft) && I.parentNode.nodeName !== "#document" && (Tt = " translate(" + r(at) + " " + r(ft) + ") "), U = Tt + " matrix(" + z + " 0 0 " + V + " " + (T * z + kt) + " " + (R * V + jt) + ") ", I.nodeName === "svg") {
        for (B = I.ownerDocument.createElementNS(a.svgNS, "g"); I.firstChild;) B.appendChild(I.firstChild);
        I.appendChild(B);
      } else (B = I).removeAttribute("x"), B.removeAttribute("y"), U = B.getAttribute("transform") + U;
      return B.setAttribute("transform", U), gt;
    }
    function W(I, T) {
      var R = "xlink:href", Q = C(I, T.getAttribute(R).substr(1));
      if (Q && Q.getAttribute(R) && W(I, Q), ["gradientTransform", "x1", "x2", "y1", "y2", "gradientUnits", "cx", "cy", "r", "fx", "fy"].forEach(function (U) {
        Q && !T.hasAttribute(U) && Q.hasAttribute(U) && T.setAttribute(U, Q.getAttribute(U));
      }), !T.children.length) for (var K = Q.cloneNode(!0); K.firstChild;) T.appendChild(K.firstChild);
      T.removeAttribute(R);
    }
    a.parseSVGDocument = function (I, T, R, Q) {
      if (I) {
        (function (at) {
          for (var ft = k(at, ["use", "svg:use"]), ot = 0; ft.length && ot < ft.length;) {
            var ct = ft[ot], st = ct.getAttribute("xlink:href") || ct.getAttribute("href");
            if (st === null) return;
            var yt, gt, Tt, kt, jt = st.substr(1), Kt = ct.getAttribute("x") || 0, Rt = ct.getAttribute("y") || 0, Bt = C(at, jt).cloneNode(!0), se = (Bt.getAttribute("transform") || "") + " translate(" + Kt + ", " + Rt + ")", zt = ft.length, ie = a.svgNS;
            if (N(Bt), /^svg$/i.test(Bt.nodeName)) {
              var fi = Bt.ownerDocument.createElementNS(ie, "g");
              for (gt = 0, kt = (Tt = Bt.attributes).length; gt < kt; gt++) yt = Tt.item(gt), fi.setAttributeNS(ie, yt.nodeName, yt.nodeValue);
              for (; Bt.firstChild;) fi.appendChild(Bt.firstChild);
              Bt = fi;
            }
            for (gt = 0, kt = (Tt = ct.attributes).length; gt < kt; gt++) (yt = Tt.item(gt)).nodeName !== "x" && yt.nodeName !== "y" && yt.nodeName !== "xlink:href" && yt.nodeName !== "href" && (yt.nodeName === "transform" ? se = yt.nodeValue + " " + se : Bt.setAttribute(yt.nodeName, yt.nodeValue));
            Bt.setAttribute("transform", se), Bt.setAttribute("instantiated_by_use", "1"), Bt.removeAttribute("id"), ct.parentNode.replaceChild(Bt, ct), ft.length === zt && ot++;
          }
        })(I);
        var K, U, B = a.Object.__uid++, Z = N(I), z = a.util.toArray(I.getElementsByTagName("*"));
        if (Z.crossOrigin = Q && Q.crossOrigin, Z.svgUid = B, z.length === 0 && a.isLikelyNode) {
          var V = [];
          for (K = 0, U = (z = I.selectNodes('//*[name(.)!="svg"]')).length; K < U; K++) V[K] = z[K];
          z = V;
        }
        var et = z.filter(function (at) {
          return N(at), a.svgValidTagNamesRegEx.test(at.nodeName.replace("svg:", "")) && !function (ft, ot) {
            for (; ft && (ft = ft.parentNode);) if (ft.nodeName && ot.test(ft.nodeName.replace("svg:", "")) && !ft.getAttribute("instantiated_by_use")) return !0;
            return !1;
          }(at, a.svgInvalidAncestorsRegEx);
        });
        if (!et || et && !et.length) T && T([], {});
        else {
          var lt = {};
          z.filter(function (at) {
            return at.nodeName.replace("svg:", "") === "clipPath";
          }).forEach(function (at) {
            var ft = at.getAttribute("id");
            lt[ft] = a.util.toArray(at.getElementsByTagName("*")).filter(function (ot) {
              return a.svgValidTagNamesRegEx.test(ot.nodeName.replace("svg:", ""));
            });
          }), a.gradientDefs[B] = a.getGradientDefs(I), a.cssRules[B] = a.getCSSRules(I), a.clipPaths[B] = lt, a.parseElements(et, function (at, ft) {
            T && (T(at, Z, ft, z), delete a.gradientDefs[B], delete a.cssRules[B], delete a.clipPaths[B]);
          }, f(Z), R, Q);
        }
      }
    };
    var X = new RegExp("(normal|italic)?\\s*(normal|small-caps)?\\s*(normal|bold|bolder|lighter|100|200|300|400|500|600|700|800|900)?\\s*(" + a.reNum + "(?:px|cm|mm|em|pt|pc|in)*)(?:\\/(normal|" + a.reNum + "))?\\s+(.*)");
    c(a, {
      parseFontDeclaration: function (I, T) {
        var R = I.match(X);
        if (R) {
          var Q = R[1], K = R[3], U = R[4], B = R[5], Z = R[6];
          Q && (T.fontStyle = Q), K && (T.fontWeight = isNaN(parseFloat(K)) ? K : parseFloat(K)), U && (T.fontSize = r(U)), Z && (T.fontFamily = Z), B && (T.lineHeight = B === "normal" ? 1 : B);
        }
      }, getGradientDefs: function (I) {
        var T, R = k(I, ["linearGradient", "radialGradient", "svg:linearGradient", "svg:radialGradient"]), Q = 0, K = {};
        for (Q = R.length; Q--;) (T = R[Q]).getAttribute("xlink:href") && W(I, T), K[T.getAttribute("id")] = T;
        return K;
      }, parseAttributes: function (I, T, R) {
        if (I) {
          var Q, K, U, B = {};
          R === void 0 && (R = I.getAttribute("svgUid")), I.parentNode && a.svgValidParentsRegEx.test(I.parentNode.nodeName) && (B = a.parseAttributes(I.parentNode, T, R));
          var Z = T.reduce(function (ot, ct) {
            return (Q = I.getAttribute(ct)) && (ot[ct] = Q), ot;
          }, {}), z = c(function (ot, ct) {
            var st = {};
            for (var yt in a.cssRules[ct]) if (D(ot, yt.split(" "))) for (var gt in a.cssRules[ct][yt]) st[gt] = a.cssRules[ct][yt][gt];
            return st;
          }(I, R), a.parseStyleAttribute(I));
          Z = c(Z, z), z[m] && I.setAttribute(m, z[m]), K = U = B.fontSize || a.Text.DEFAULT_SVG_FONT_SIZE, Z[p] && (Z[p] = K = r(Z[p], U));
          var V, et, lt = {};
          for (var at in Z) et = x(V = y(at), Z[at], B, K), lt[V] = et;
          lt && lt.font && a.parseFontDeclaration(lt.font, lt);
          var ft = c(B, lt);
          return a.svgValidParentsRegEx.test(I.nodeName) ? ft : function (ot) {
            for (var ct in h) if (ot[h[ct]] !== void 0 && ot[ct] !== "") {
              if (ot[ct] === void 0) {
                if (!a.Object.prototype[ct]) continue;
                ot[ct] = a.Object.prototype[ct];
              }
              if (ot[ct].indexOf("url(") !== 0) {
                var st = new a.Color(ot[ct]);
                ot[ct] = st.setAlpha(i(st.getAlpha() * ot[h[ct]], 2)).toRgba();
              }
            }
            return ot;
          }(ft);
        }
      }, parseElements: function (I, T, R, Q, K) {
        new a.ElementsParser(I, T, R, Q, K).parse();
      }, parseStyleAttribute: function (I) {
        var T = {}, R = I.getAttribute("style");
        return R && (typeof R == "string" ? function (Q, K) {
          var U, B;
          Q.replace(/;\s*$/, "").split(";").forEach(function (Z) {
            var z = Z.split(":");
            U = z[0].trim().toLowerCase(), B = z[1].trim(), K[U] = B;
          });
        }(R, T) : function (Q, K) {
          var U, B;
          for (var Z in Q) Q[Z] !== void 0 && (U = Z.toLowerCase(), B = Q[Z], K[U] = B);
        }(R, T)), T;
      }, parsePointsAttribute: function (I) {
        if (!I) return null;
        var T, R, Q = [];
        for (T = 0, R = (I = (I = I.replace(/,/g, " ").trim()).split(/\s+/)).length; T < R; T += 2) Q.push({ x: parseFloat(I[T]), y: parseFloat(I[T + 1]) });
        return Q;
      }, getCSSRules: function (I) {
        var T, R, Q = I.getElementsByTagName("style"), K = {};
        for (T = 0, R = Q.length; T < R; T++) {
          var U = Q[T].textContent;
          (U = U.replace(/\/\*[\s\S]*?\*\//g, "")).trim() !== "" && U.match(/[^{]*\{[\s\S]*?\}/g).map(function (B) {
            return B.trim();
          }).forEach(function (B) {
            var Z = B.match(/([\s\S]*?)\s*\{([^}]*)\}/), z = {}, V = Z[2].trim().replace(/;$/, "").split(/\s*;\s*/);
            for (T = 0, R = V.length; T < R; T++) {
              var et = V[T].split(/\s*:\s*/), lt = et[0], at = et[1];
              z[lt] = at;
            }
            (B = Z[1]).split(",").forEach(function (ft) {
              (ft = ft.replace(/^svg/i, "").trim()) !== "" && (K[ft] ? a.util.object.extend(K[ft], z) : K[ft] = a.util.object.clone(z));
            });
          });
        }
        return K;
      }, loadSVGFromURL: function (I, T, R, Q) {
        I = I.replace(/^\n\s*/, "").trim(), new a.util.request(I, {
          method: "get", onComplete: function (K) {
            var U = K.responseXML;
            if (!U || !U.documentElement) return T && T(null), !1;
            a.parseSVGDocument(U.documentElement, function (B, Z, z, V) {
              T && T(B, Z, z, V);
            }, R, Q);
          }
        });
      }, loadSVGFromString: function (I, T, R, Q) {
        var K = new a.window.DOMParser().parseFromString(I.trim(), "text/xml");
        a.parseSVGDocument(K.documentElement, function (U, B, Z, z) {
          T(U, B, Z, z);
        }, R, Q);
      }
    });
  }(t), v.ElementsParser = function (l, a, c, f, i, r) {
    this.elements = l, this.callback = a, this.options = c, this.reviver = f, this.svgUid = c && c.svgUid || 0, this.parsingOptions = i, this.regexUrl = /^url\(['"]?#([^'"]+)['"]?\)/g, this.doc = r;
  }, (Y = v.ElementsParser.prototype).parse = function () {
    this.instances = new Array(this.elements.length), this.numElements = this.elements.length, this.createObjects();
  }, Y.createObjects = function () {
    var l = this;
    this.elements.forEach(function (a, c) {
      a.setAttribute("svgUid", l.svgUid), l.createObject(a, c);
    });
  }, Y.findTag = function (l) {
    return v[v.util.string.capitalize(l.tagName.replace("svg:", ""))];
  }, Y.createObject = function (l, a) {
    var c = this.findTag(l);
    if (c && c.fromElement) try {
      c.fromElement(l, this.createCallback(a, l), this.options);
    } catch (f) {
      v.log(f);
    }
    else this.checkIfDone();
  }, Y.createCallback = function (l, a) {
    var c = this;
    return function (f) {
      var i;
      c.resolveGradient(f, a, "fill"), c.resolveGradient(f, a, "stroke"), f instanceof v.Image && f._originalElement && (i = f.parsePreserveAspectRatioAttribute(a)), f._removeTransformMatrix(i), c.resolveClipPath(f, a), c.reviver && c.reviver(a, f), c.instances[l] = f, c.checkIfDone();
    };
  }, Y.extractPropertyDefinition = function (l, a, c) {
    var f = l[a], i = this.regexUrl;
    if (i.test(f)) {
      i.lastIndex = 0;
      var r = i.exec(f)[1];
      return i.lastIndex = 0, v[c][this.svgUid][r];
    }
  }, Y.resolveGradient = function (l, a, c) {
    var f = this.extractPropertyDefinition(l, c, "gradientDefs");
    if (f) {
      var i = a.getAttribute(c + "-opacity"), r = v.Gradient.fromElement(f, l, i, this.options);
      l.set(c, r);
    }
  }, Y.createClipPathCallback = function (l, a) {
    return function (c) {
      c._removeTransformMatrix(), c.fillRule = c.clipRule, a.push(c);
    };
  }, Y.resolveClipPath = function (l, a) {
    var c, f, i, r, s = this.extractPropertyDefinition(l, "clipPath", "clipPaths");
    if (s) {
      i = [], f = v.util.invertTransform(l.calcTransformMatrix());
      for (var u = s[0].parentNode, h = a; h.parentNode && h.getAttribute("clip-path") !== l.clipPath;) h = h.parentNode;
      h.parentNode.appendChild(u);
      for (var p = 0; p < s.length; p++) c = s[p], this.findTag(c).fromElement(c, this.createClipPathCallback(l, i), this.options);
      s = i.length === 1 ? i[0] : new v.Group(i), r = v.util.multiplyTransformMatrices(f, s.calcTransformMatrix()), s.clipPath && this.resolveClipPath(s, h);
      var m = v.util.qrDecompose(r);
      s.flipX = !1, s.flipY = !1, s.set("scaleX", m.scaleX), s.set("scaleY", m.scaleY), s.angle = m.angle, s.skewX = m.skewX, s.skewY = 0, s.setPositionByOrigin({ x: m.translateX, y: m.translateY }, "center", "center"), l.clipPath = s;
    } else delete l.clipPath;
  }, Y.checkIfDone = function () {
    --this.numElements == 0 && (this.instances = this.instances.filter(function (l) {
      return l != null;
    }), this.callback(this.instances, this.elements));
  }, function (l) {
    var a = l.fabric || (l.fabric = {});
    function c(f, i) {
      this.x = f, this.y = i;
    }
    a.Point ? a.warn("fabric.Point is already defined") : (a.Point = c, c.prototype = {
      type: "point", constructor: c, add: function (f) {
        return new c(this.x + f.x, this.y + f.y);
      }, addEquals: function (f) {
        return this.x += f.x, this.y += f.y, this;
      }, scalarAdd: function (f) {
        return new c(this.x + f, this.y + f);
      }, scalarAddEquals: function (f) {
        return this.x += f, this.y += f, this;
      }, subtract: function (f) {
        return new c(this.x - f.x, this.y - f.y);
      }, subtractEquals: function (f) {
        return this.x -= f.x, this.y -= f.y, this;
      }, scalarSubtract: function (f) {
        return new c(this.x - f, this.y - f);
      }, scalarSubtractEquals: function (f) {
        return this.x -= f, this.y -= f, this;
      }, multiply: function (f) {
        return new c(this.x * f, this.y * f);
      }, multiplyEquals: function (f) {
        return this.x *= f, this.y *= f, this;
      }, divide: function (f) {
        return new c(this.x / f, this.y / f);
      }, divideEquals: function (f) {
        return this.x /= f, this.y /= f, this;
      }, eq: function (f) {
        return this.x === f.x && this.y === f.y;
      }, lt: function (f) {
        return this.x < f.x && this.y < f.y;
      }, lte: function (f) {
        return this.x <= f.x && this.y <= f.y;
      }, gt: function (f) {
        return this.x > f.x && this.y > f.y;
      }, gte: function (f) {
        return this.x >= f.x && this.y >= f.y;
      }, lerp: function (f, i) {
        return i === void 0 && (i = 0.5), i = Math.max(Math.min(1, i), 0), new c(this.x + (f.x - this.x) * i, this.y + (f.y - this.y) * i);
      }, distanceFrom: function (f) {
        var i = this.x - f.x, r = this.y - f.y;
        return Math.sqrt(i * i + r * r);
      }, midPointFrom: function (f) {
        return this.lerp(f);
      }, min: function (f) {
        return new c(Math.min(this.x, f.x), Math.min(this.y, f.y));
      }, max: function (f) {
        return new c(Math.max(this.x, f.x), Math.max(this.y, f.y));
      }, toString: function () {
        return this.x + "," + this.y;
      }, setXY: function (f, i) {
        return this.x = f, this.y = i, this;
      }, setX: function (f) {
        return this.x = f, this;
      }, setY: function (f) {
        return this.y = f, this;
      }, setFromPoint: function (f) {
        return this.x = f.x, this.y = f.y, this;
      }, swap: function (f) {
        var i = this.x, r = this.y;
        this.x = f.x, this.y = f.y, f.x = i, f.y = r;
      }, clone: function () {
        return new c(this.x, this.y);
      }
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {});
    function c(f) {
      this.status = f, this.points = [];
    }
    a.Intersection ? a.warn("fabric.Intersection is already defined") : (a.Intersection = c, a.Intersection.prototype = {
      constructor: c, appendPoint: function (f) {
        return this.points.push(f), this;
      }, appendPoints: function (f) {
        return this.points = this.points.concat(f), this;
      }
    }, a.Intersection.intersectLineLine = function (f, i, r, s) {
      var u, h = (s.x - r.x) * (f.y - r.y) - (s.y - r.y) * (f.x - r.x), p = (i.x - f.x) * (f.y - r.y) - (i.y - f.y) * (f.x - r.x), m = (s.y - r.y) * (i.x - f.x) - (s.x - r.x) * (i.y - f.y);
      if (m !== 0) {
        var y = h / m, x = p / m;
        0 <= y && y <= 1 && 0 <= x && x <= 1 ? (u = new c("Intersection")).appendPoint(new a.Point(f.x + y * (i.x - f.x), f.y + y * (i.y - f.y))) : u = new c();
      } else u = new c(h === 0 || p === 0 ? "Coincident" : "Parallel");
      return u;
    }, a.Intersection.intersectLinePolygon = function (f, i, r) {
      var s, u, h, p, m = new c(), y = r.length;
      for (p = 0; p < y; p++) s = r[p], u = r[(p + 1) % y], h = c.intersectLineLine(f, i, s, u), m.appendPoints(h.points);
      return m.points.length > 0 && (m.status = "Intersection"), m;
    }, a.Intersection.intersectPolygonPolygon = function (f, i) {
      var r, s = new c(), u = f.length;
      for (r = 0; r < u; r++) {
        var h = f[r], p = f[(r + 1) % u], m = c.intersectLinePolygon(h, p, i);
        s.appendPoints(m.points);
      }
      return s.points.length > 0 && (s.status = "Intersection"), s;
    }, a.Intersection.intersectPolygonRectangle = function (f, i, r) {
      var s = i.min(r), u = i.max(r), h = new a.Point(u.x, s.y), p = new a.Point(s.x, u.y), m = c.intersectLinePolygon(s, h, f), y = c.intersectLinePolygon(h, u, f), x = c.intersectLinePolygon(u, p, f), S = c.intersectLinePolygon(p, s, f), k = new c();
      return k.appendPoints(m.points), k.appendPoints(y.points), k.appendPoints(x.points), k.appendPoints(S.points), k.points.length > 0 && (k.status = "Intersection"), k;
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {});
    function c(i) {
      i ? this._tryParsingColor(i) : this.setSource([0, 0, 0, 1]);
    }
    function f(i, r, s) {
      return s < 0 && (s += 1), s > 1 && (s -= 1), s < 1 / 6 ? i + 6 * (r - i) * s : s < 0.5 ? r : s < 2 / 3 ? i + (r - i) * (2 / 3 - s) * 6 : i;
    }
    a.Color ? a.warn("fabric.Color is already defined.") : (a.Color = c, a.Color.prototype = {
      _tryParsingColor: function (i) {
        var r;
        i in c.colorNameMap && (i = c.colorNameMap[i]), i === "transparent" && (r = [255, 255, 255, 0]), r || (r = c.sourceFromHex(i)), r || (r = c.sourceFromRgb(i)), r || (r = c.sourceFromHsl(i)), r || (r = [0, 0, 0, 1]), r && this.setSource(r);
      }, _rgbToHsl: function (i, r, s) {
        i /= 255, r /= 255, s /= 255;
        var u, h, p, m = a.util.array.max([i, r, s]), y = a.util.array.min([i, r, s]);
        if (p = (m + y) / 2, m === y) u = h = 0;
        else {
          var x = m - y;
          switch (h = p > 0.5 ? x / (2 - m - y) : x / (m + y), m) {
            case i:
              u = (r - s) / x + (r < s ? 6 : 0);
              break;
            case r:
              u = (s - i) / x + 2;
              break;
            case s:
              u = (i - r) / x + 4;
          }
          u /= 6;
        }
        return [Math.round(360 * u), Math.round(100 * h), Math.round(100 * p)];
      }, getSource: function () {
        return this._source;
      }, setSource: function (i) {
        this._source = i;
      }, toRgb: function () {
        var i = this.getSource();
        return "rgb(" + i[0] + "," + i[1] + "," + i[2] + ")";
      }, toRgba: function () {
        var i = this.getSource();
        return "rgba(" + i[0] + "," + i[1] + "," + i[2] + "," + i[3] + ")";
      }, toHsl: function () {
        var i = this.getSource(), r = this._rgbToHsl(i[0], i[1], i[2]);
        return "hsl(" + r[0] + "," + r[1] + "%," + r[2] + "%)";
      }, toHsla: function () {
        var i = this.getSource(), r = this._rgbToHsl(i[0], i[1], i[2]);
        return "hsla(" + r[0] + "," + r[1] + "%," + r[2] + "%," + i[3] + ")";
      }, toHex: function () {
        var i, r, s, u = this.getSource();
        return i = (i = u[0].toString(16)).length === 1 ? "0" + i : i, r = (r = u[1].toString(16)).length === 1 ? "0" + r : r, s = (s = u[2].toString(16)).length === 1 ? "0" + s : s, i.toUpperCase() + r.toUpperCase() + s.toUpperCase();
      }, toHexa: function () {
        var i, r = this.getSource();
        return i = (i = (i = Math.round(255 * r[3])).toString(16)).length === 1 ? "0" + i : i, this.toHex() + i.toUpperCase();
      }, getAlpha: function () {
        return this.getSource()[3];
      }, setAlpha: function (i) {
        var r = this.getSource();
        return r[3] = i, this.setSource(r), this;
      }, toGrayscale: function () {
        var i = this.getSource(), r = parseInt((0.3 * i[0] + 0.59 * i[1] + 0.11 * i[2]).toFixed(0), 10), s = i[3];
        return this.setSource([r, r, r, s]), this;
      }, toBlackWhite: function (i) {
        var r = this.getSource(), s = (0.3 * r[0] + 0.59 * r[1] + 0.11 * r[2]).toFixed(0), u = r[3];
        return i = i || 127, s = Number(s) < Number(i) ? 0 : 255, this.setSource([s, s, s, u]), this;
      }, overlayWith: function (i) {
        i instanceof c || (i = new c(i));
        var r, s = [], u = this.getAlpha(), h = this.getSource(), p = i.getSource();
        for (r = 0; r < 3; r++) s.push(Math.round(0.5 * h[r] + 0.5 * p[r]));
        return s[3] = u, this.setSource(s), this;
      }
    }, a.Color.reRGBa = /^rgba?\(\s*(\d{1,3}(?:\.\d+)?\%?)\s*,\s*(\d{1,3}(?:\.\d+)?\%?)\s*,\s*(\d{1,3}(?:\.\d+)?\%?)\s*(?:\s*,\s*((?:\d*\.?\d+)?)\s*)?\)$/i, a.Color.reHSLa = /^hsla?\(\s*(\d{1,3})\s*,\s*(\d{1,3}\%)\s*,\s*(\d{1,3}\%)\s*(?:\s*,\s*(\d+(?:\.\d+)?)\s*)?\)$/i, a.Color.reHex = /^#?([0-9a-f]{8}|[0-9a-f]{6}|[0-9a-f]{4}|[0-9a-f]{3})$/i, a.Color.colorNameMap = { aliceblue: "#F0F8FF", antiquewhite: "#FAEBD7", aqua: "#00FFFF", aquamarine: "#7FFFD4", azure: "#F0FFFF", beige: "#F5F5DC", bisque: "#FFE4C4", black: "#000000", blanchedalmond: "#FFEBCD", blue: "#0000FF", blueviolet: "#8A2BE2", brown: "#A52A2A", burlywood: "#DEB887", cadetblue: "#5F9EA0", chartreuse: "#7FFF00", chocolate: "#D2691E", coral: "#FF7F50", cornflowerblue: "#6495ED", cornsilk: "#FFF8DC", crimson: "#DC143C", cyan: "#00FFFF", darkblue: "#00008B", darkcyan: "#008B8B", darkgoldenrod: "#B8860B", darkgray: "#A9A9A9", darkgrey: "#A9A9A9", darkgreen: "#006400", darkkhaki: "#BDB76B", darkmagenta: "#8B008B", darkolivegreen: "#556B2F", darkorange: "#FF8C00", darkorchid: "#9932CC", darkred: "#8B0000", darksalmon: "#E9967A", darkseagreen: "#8FBC8F", darkslateblue: "#483D8B", darkslategray: "#2F4F4F", darkslategrey: "#2F4F4F", darkturquoise: "#00CED1", darkviolet: "#9400D3", deeppink: "#FF1493", deepskyblue: "#00BFFF", dimgray: "#696969", dimgrey: "#696969", dodgerblue: "#1E90FF", firebrick: "#B22222", floralwhite: "#FFFAF0", forestgreen: "#228B22", fuchsia: "#FF00FF", gainsboro: "#DCDCDC", ghostwhite: "#F8F8FF", gold: "#FFD700", goldenrod: "#DAA520", gray: "#808080", grey: "#808080", green: "#008000", greenyellow: "#ADFF2F", honeydew: "#F0FFF0", hotpink: "#FF69B4", indianred: "#CD5C5C", indigo: "#4B0082", ivory: "#FFFFF0", khaki: "#F0E68C", lavender: "#E6E6FA", lavenderblush: "#FFF0F5", lawngreen: "#7CFC00", lemonchiffon: "#FFFACD", lightblue: "#ADD8E6", lightcoral: "#F08080", lightcyan: "#E0FFFF", lightgoldenrodyellow: "#FAFAD2", lightgray: "#D3D3D3", lightgrey: "#D3D3D3", lightgreen: "#90EE90", lightpink: "#FFB6C1", lightsalmon: "#FFA07A", lightseagreen: "#20B2AA", lightskyblue: "#87CEFA", lightslategray: "#778899", lightslategrey: "#778899", lightsteelblue: "#B0C4DE", lightyellow: "#FFFFE0", lime: "#00FF00", limegreen: "#32CD32", linen: "#FAF0E6", magenta: "#FF00FF", maroon: "#800000", mediumaquamarine: "#66CDAA", mediumblue: "#0000CD", mediumorchid: "#BA55D3", mediumpurple: "#9370DB", mediumseagreen: "#3CB371", mediumslateblue: "#7B68EE", mediumspringgreen: "#00FA9A", mediumturquoise: "#48D1CC", mediumvioletred: "#C71585", midnightblue: "#191970", mintcream: "#F5FFFA", mistyrose: "#FFE4E1", moccasin: "#FFE4B5", navajowhite: "#FFDEAD", navy: "#000080", oldlace: "#FDF5E6", olive: "#808000", olivedrab: "#6B8E23", orange: "#FFA500", orangered: "#FF4500", orchid: "#DA70D6", palegoldenrod: "#EEE8AA", palegreen: "#98FB98", paleturquoise: "#AFEEEE", palevioletred: "#DB7093", papayawhip: "#FFEFD5", peachpuff: "#FFDAB9", peru: "#CD853F", pink: "#FFC0CB", plum: "#DDA0DD", powderblue: "#B0E0E6", purple: "#800080", rebeccapurple: "#663399", red: "#FF0000", rosybrown: "#BC8F8F", royalblue: "#4169E1", saddlebrown: "#8B4513", salmon: "#FA8072", sandybrown: "#F4A460", seagreen: "#2E8B57", seashell: "#FFF5EE", sienna: "#A0522D", silver: "#C0C0C0", skyblue: "#87CEEB", slateblue: "#6A5ACD", slategray: "#708090", slategrey: "#708090", snow: "#FFFAFA", springgreen: "#00FF7F", steelblue: "#4682B4", tan: "#D2B48C", teal: "#008080", thistle: "#D8BFD8", tomato: "#FF6347", turquoise: "#40E0D0", violet: "#EE82EE", wheat: "#F5DEB3", white: "#FFFFFF", whitesmoke: "#F5F5F5", yellow: "#FFFF00", yellowgreen: "#9ACD32" }, a.Color.fromRgb = function (i) {
      return c.fromSource(c.sourceFromRgb(i));
    }, a.Color.sourceFromRgb = function (i) {
      var r = i.match(c.reRGBa);
      if (r) {
        var s = parseInt(r[1], 10) / (/%$/.test(r[1]) ? 100 : 1) * (/%$/.test(r[1]) ? 255 : 1), u = parseInt(r[2], 10) / (/%$/.test(r[2]) ? 100 : 1) * (/%$/.test(r[2]) ? 255 : 1), h = parseInt(r[3], 10) / (/%$/.test(r[3]) ? 100 : 1) * (/%$/.test(r[3]) ? 255 : 1);
        return [parseInt(s, 10), parseInt(u, 10), parseInt(h, 10), r[4] ? parseFloat(r[4]) : 1];
      }
    }, a.Color.fromRgba = c.fromRgb, a.Color.fromHsl = function (i) {
      return c.fromSource(c.sourceFromHsl(i));
    }, a.Color.sourceFromHsl = function (i) {
      var r = i.match(c.reHSLa);
      if (r) {
        var s, u, h, p = (parseFloat(r[1]) % 360 + 360) % 360 / 360, m = parseFloat(r[2]) / (/%$/.test(r[2]) ? 100 : 1), y = parseFloat(r[3]) / (/%$/.test(r[3]) ? 100 : 1);
        if (m === 0) s = u = h = y;
        else {
          var x = y <= 0.5 ? y * (m + 1) : y + m - y * m, S = 2 * y - x;
          s = f(S, x, p + 1 / 3), u = f(S, x, p), h = f(S, x, p - 1 / 3);
        }
        return [Math.round(255 * s), Math.round(255 * u), Math.round(255 * h), r[4] ? parseFloat(r[4]) : 1];
      }
    }, a.Color.fromHsla = c.fromHsl, a.Color.fromHex = function (i) {
      return c.fromSource(c.sourceFromHex(i));
    }, a.Color.sourceFromHex = function (i) {
      if (i.match(c.reHex)) {
        var r = i.slice(i.indexOf("#") + 1), s = r.length === 3 || r.length === 4, u = r.length === 8 || r.length === 4, h = s ? r.charAt(0) + r.charAt(0) : r.substring(0, 2), p = s ? r.charAt(1) + r.charAt(1) : r.substring(2, 4), m = s ? r.charAt(2) + r.charAt(2) : r.substring(4, 6), y = u ? s ? r.charAt(3) + r.charAt(3) : r.substring(6, 8) : "FF";
        return [parseInt(h, 16), parseInt(p, 16), parseInt(m, 16), parseFloat((parseInt(y, 16) / 255).toFixed(2))];
      }
    }, a.Color.fromSource = function (i) {
      var r = new c();
      return r.setSource(i), r;
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = ["e", "se", "s", "sw", "w", "nw", "n", "ne", "e"], f = ["ns", "nesw", "ew", "nwse"], i = {}, r = "left", s = "top", u = "right", h = "bottom", p = "center", m = { top: h, bottom: s, left: u, right: r, center: p }, y = a.util.radiansToDegrees, x = Math.sign || function (U) {
      return (U > 0) - (U < 0) || +U;
    };
    function S(U, B) {
      var Z = U.angle + y(Math.atan2(B.y, B.x)) + 360;
      return Math.round(Z % 360 / 45);
    }
    function k(U, B) {
      var Z = B.transform.target, z = Z.canvas, V = a.util.object.clone(B);
      V.target = Z, z && z.fire("object:" + U, V), Z.fire(U, B);
    }
    function D(U, B) {
      var Z = B.canvas, z = U[Z.uniScaleKey];
      return Z.uniformScaling && !z || !Z.uniformScaling && z;
    }
    function M(U) {
      return U.originX === p && U.originY === p;
    }
    function C(U, B, Z) {
      var z = U.lockScalingX, V = U.lockScalingY;
      return !(!z || !V) || !(B || !z && !V || !Z) || !(!z || B !== "x") || !(!V || B !== "y");
    }
    function O(U, B, Z, z) {
      return { e: U, transform: B, pointer: { x: Z, y: z } };
    }
    function N(U) {
      return function (B, Z, z, V) {
        var et = Z.target, lt = et.getCenterPoint(), at = et.translateToOriginPoint(lt, Z.originX, Z.originY), ft = U(B, Z, z, V);
        return et.setPositionByOrigin(at, Z.originX, Z.originY), ft;
      };
    }
    function W(U, B) {
      return function (Z, z, V, et) {
        var lt = B(Z, z, V, et);
        return lt && k(U, O(Z, z, V, et)), lt;
      };
    }
    function X(U, B, Z, z, V) {
      var et = U.target, lt = et.controls[U.corner], at = et.canvas.getZoom(), ft = et.padding / at, ot = et.toLocalPoint(new a.Point(z, V), B, Z);
      return ot.x >= ft && (ot.x -= ft), ot.x <= -ft && (ot.x += ft), ot.y >= ft && (ot.y -= ft), ot.y <= ft && (ot.y += ft), ot.x -= lt.offsetX, ot.y -= lt.offsetY, ot;
    }
    function I(U) {
      return U.flipX !== U.flipY;
    }
    function T(U, B, Z, z, V) {
      if (U[B] !== 0) {
        var et = V / U._getTransformedDimensions()[z] * U[Z];
        U.set(Z, et);
      }
    }
    function R(U, B, Z, z) {
      var V, et = B.target, lt = et._getTransformedDimensions(0, et.skewY), at = X(B, B.originX, B.originY, Z, z), ft = Math.abs(2 * at.x) - lt.x, ot = et.skewX;
      ft < 2 ? V = 0 : (V = y(Math.atan2(ft / et.scaleX, lt.y / et.scaleY)), B.originX === r && B.originY === h && (V = -V), B.originX === u && B.originY === s && (V = -V), I(et) && (V = -V));
      var ct = ot !== V;
      if (ct) {
        var st = et._getTransformedDimensions().y;
        et.set("skewX", V), T(et, "skewY", "scaleY", "y", st);
      }
      return ct;
    }
    function Q(U, B, Z, z) {
      var V, et = B.target, lt = et._getTransformedDimensions(et.skewX, 0), at = X(B, B.originX, B.originY, Z, z), ft = Math.abs(2 * at.y) - lt.y, ot = et.skewY;
      ft < 2 ? V = 0 : (V = y(Math.atan2(ft / et.scaleY, lt.x / et.scaleX)), B.originX === r && B.originY === h && (V = -V), B.originX === u && B.originY === s && (V = -V), I(et) && (V = -V));
      var ct = ot !== V;
      if (ct) {
        var st = et._getTransformedDimensions().x;
        et.set("skewY", V), T(et, "skewX", "scaleX", "x", st);
      }
      return ct;
    }
    function K(U, B, Z, z, V) {
      V = V || {};
      var et, lt, at, ft, ot, ct, st = B.target, yt = st.lockScalingX, gt = st.lockScalingY, Tt = V.by, kt = D(U, st), jt = C(st, Tt, kt), Kt = B.gestureScale;
      if (jt) return !1;
      if (Kt) lt = B.scaleX * Kt, at = B.scaleY * Kt;
      else {
        if (et = X(B, B.originX, B.originY, Z, z), ot = Tt !== "y" ? x(et.x) : 1, ct = Tt !== "x" ? x(et.y) : 1, B.signX || (B.signX = ot), B.signY || (B.signY = ct), st.lockScalingFlip && (B.signX !== ot || B.signY !== ct)) return !1;
        if (ft = st._getTransformedDimensions(), kt && !Tt) {
          var Rt = Math.abs(et.x) + Math.abs(et.y), Bt = B.original, se = Rt / (Math.abs(ft.x * Bt.scaleX / st.scaleX) + Math.abs(ft.y * Bt.scaleY / st.scaleY));
          lt = Bt.scaleX * se, at = Bt.scaleY * se;
        } else lt = Math.abs(et.x * st.scaleX / ft.x), at = Math.abs(et.y * st.scaleY / ft.y);
        M(B) && (lt *= 2, at *= 2), B.signX !== ot && Tt !== "y" && (B.originX = m[B.originX], lt *= -1, B.signX = ot), B.signY !== ct && Tt !== "x" && (B.originY = m[B.originY], at *= -1, B.signY = ct);
      }
      var zt = st.scaleX, ie = st.scaleY;
      return Tt ? (Tt === "x" && st.set("scaleX", lt), Tt === "y" && st.set("scaleY", at)) : (!yt && st.set("scaleX", lt), !gt && st.set("scaleY", at)), zt !== st.scaleX || ie !== st.scaleY;
    }
    i.scaleCursorStyleHandler = function (U, B, Z) {
      var z = D(U, Z), V = "";
      if (B.x !== 0 && B.y === 0 ? V = "x" : B.x === 0 && B.y !== 0 && (V = "y"), C(Z, V, z)) return "not-allowed";
      var et = S(Z, B);
      return c[et] + "-resize";
    }, i.skewCursorStyleHandler = function (U, B, Z) {
      var z = "not-allowed";
      if (B.x !== 0 && Z.lockSkewingY || B.y !== 0 && Z.lockSkewingX) return z;
      var V = S(Z, B) % 4;
      return f[V] + "-resize";
    }, i.scaleSkewCursorStyleHandler = function (U, B, Z) {
      return U[Z.canvas.altActionKey] ? i.skewCursorStyleHandler(U, B, Z) : i.scaleCursorStyleHandler(U, B, Z);
    }, i.rotationWithSnapping = W("rotating", N(function (U, B, Z, z) {
      var V = B, et = V.target, lt = et.translateToOriginPoint(et.getCenterPoint(), V.originX, V.originY);
      if (et.lockRotation) return !1;
      var at, ft = Math.atan2(V.ey - lt.y, V.ex - lt.x), ot = Math.atan2(z - lt.y, Z - lt.x), ct = y(ot - ft + V.theta);
      if (et.snapAngle > 0) {
        var st = et.snapAngle, yt = et.snapThreshold || st, gt = Math.ceil(ct / st) * st, Tt = Math.floor(ct / st) * st;
        Math.abs(ct - Tt) < yt ? ct = Tt : Math.abs(ct - gt) < yt && (ct = gt);
      }
      return ct < 0 && (ct = 360 + ct), ct %= 360, at = et.angle !== ct, et.angle = ct, at;
    })), i.scalingEqually = W("scaling", N(function (U, B, Z, z) {
      return K(U, B, Z, z);
    })), i.scalingX = W("scaling", N(function (U, B, Z, z) {
      return K(U, B, Z, z, { by: "x" });
    })), i.scalingY = W("scaling", N(function (U, B, Z, z) {
      return K(U, B, Z, z, { by: "y" });
    })), i.scalingYOrSkewingX = function (U, B, Z, z) {
      return U[B.target.canvas.altActionKey] ? i.skewHandlerX(U, B, Z, z) : i.scalingY(U, B, Z, z);
    }, i.scalingXOrSkewingY = function (U, B, Z, z) {
      return U[B.target.canvas.altActionKey] ? i.skewHandlerY(U, B, Z, z) : i.scalingX(U, B, Z, z);
    }, i.changeWidth = W("resizing", N(function (U, B, Z, z) {
      var V = B.target, et = X(B, B.originX, B.originY, Z, z), lt = V.strokeWidth / (V.strokeUniform ? V.scaleX : 1), at = M(B) ? 2 : 1, ft = V.width, ot = Math.abs(et.x * at / V.scaleX) - lt;
      return V.set("width", Math.max(ot, 0)), ft !== ot;
    })), i.skewHandlerX = function (U, B, Z, z) {
      var V, et = B.target, lt = et.skewX, at = B.originY;
      return !et.lockSkewingX && (lt === 0 ? V = X(B, p, p, Z, z).x > 0 ? r : u : (lt > 0 && (V = at === s ? r : u), lt < 0 && (V = at === s ? u : r), I(et) && (V = V === r ? u : r)), B.originX = V, W("skewing", N(R))(U, B, Z, z));
    }, i.skewHandlerY = function (U, B, Z, z) {
      var V, et = B.target, lt = et.skewY, at = B.originX;
      return !et.lockSkewingY && (lt === 0 ? V = X(B, p, p, Z, z).y > 0 ? s : h : (lt > 0 && (V = at === r ? s : h), lt < 0 && (V = at === r ? h : s), I(et) && (V = V === s ? h : s)), B.originY = V, W("skewing", N(Q))(U, B, Z, z));
    }, i.dragHandler = function (U, B, Z, z) {
      var V = B.target, et = Z - B.offsetX, lt = z - B.offsetY, at = !V.get("lockMovementX") && V.left !== et, ft = !V.get("lockMovementY") && V.top !== lt;
      return at && V.set("left", et), ft && V.set("top", lt), (at || ft) && k("moving", O(U, B, Z, z)), at || ft;
    }, i.scaleOrSkewActionName = function (U, B, Z) {
      var z = U[Z.canvas.altActionKey];
      return B.x === 0 ? z ? "skewX" : "scaleY" : B.y === 0 ? z ? "skewY" : "scaleX" : void 0;
    }, i.rotationStyleHandler = function (U, B, Z) {
      return Z.lockRotation ? "not-allowed" : B.cursorStyle;
    }, i.fireEvent = k, i.wrapWithFixedAnchor = N, i.wrapWithFireEvent = W, i.getLocalPoint = X, a.controlsUtils = i;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.degreesToRadians, f = a.controlsUtils;
    f.renderCircleControl = function (i, r, s, u, h) {
      u = u || {};
      var p, m = this.sizeX || u.cornerSize || h.cornerSize, y = this.sizeY || u.cornerSize || h.cornerSize, x = u.transparentCorners !== void 0 ? u.transparentCorners : h.transparentCorners, S = x ? "stroke" : "fill", k = !x && (u.cornerStrokeColor || h.cornerStrokeColor), D = r, M = s;
      i.save(), i.fillStyle = u.cornerColor || h.cornerColor, i.strokeStyle = u.cornerStrokeColor || h.cornerStrokeColor, m > y ? (p = m, i.scale(1, y / m), M = s * m / y) : y > m ? (p = y, i.scale(m / y, 1), D = r * y / m) : p = m, i.lineWidth = 1, i.beginPath(), i.arc(D, M, p / 2, 0, 2 * Math.PI, !1), i[S](), k && i.stroke(), i.restore();
    }, f.renderSquareControl = function (i, r, s, u, h) {
      u = u || {};
      var p = this.sizeX || u.cornerSize || h.cornerSize, m = this.sizeY || u.cornerSize || h.cornerSize, y = u.transparentCorners !== void 0 ? u.transparentCorners : h.transparentCorners, x = y ? "stroke" : "fill", S = !y && (u.cornerStrokeColor || h.cornerStrokeColor), k = p / 2, D = m / 2;
      i.save(), i.fillStyle = u.cornerColor || h.cornerColor, i.strokeStyle = u.cornerStrokeColor || h.cornerStrokeColor, i.lineWidth = 1, i.translate(r, s), i.rotate(c(h.angle)), i[x + "Rect"](-k, -D, p, m), S && i.strokeRect(-k, -D, p, m), i.restore();
    };
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {});
    a.Control = function (c) {
      for (var f in c) this[f] = c[f];
    }, a.Control.prototype = {
      visible: !0, actionName: "scale", angle: 0, x: 0, y: 0, offsetX: 0, offsetY: 0, sizeX: null, sizeY: null, touchSizeX: null, touchSizeY: null, cursorStyle: "crosshair", withConnection: !1, actionHandler: function () {
      }, mouseDownHandler: function () {
      }, mouseUpHandler: function () {
      }, getActionHandler: function () {
        return this.actionHandler;
      }, getMouseDownHandler: function () {
        return this.mouseDownHandler;
      }, getMouseUpHandler: function () {
        return this.mouseUpHandler;
      }, cursorStyleHandler: function (c, f) {
        return f.cursorStyle;
      }, getActionName: function (c, f) {
        return f.actionName;
      }, getVisibility: function (c, f) {
        var i = c._controlsVisibility;
        return i && i[f] !== void 0 ? i[f] : this.visible;
      }, setVisibility: function (c) {
        this.visible = c;
      }, positionHandler: function (c, f) {
        return a.util.transformPoint({ x: this.x * c.x + this.offsetX, y: this.y * c.y + this.offsetY }, f);
      }, calcCornerCoords: function (c, f, i, r, s) {
        var u, h, p, m, y = s ? this.touchSizeX : this.sizeX, x = s ? this.touchSizeY : this.sizeY;
        if (y && x && y !== x) {
          var S = Math.atan2(x, y), k = Math.sqrt(y * y + x * x) / 2, D = S - a.util.degreesToRadians(c), M = Math.PI / 2 - S - a.util.degreesToRadians(c);
          u = k * a.util.cos(D), h = k * a.util.sin(D), p = k * a.util.cos(M), m = k * a.util.sin(M);
        } else
          k = 0.7071067812 * (y && x ? y : f), D = a.util.degreesToRadians(45 - c), u = p = k * a.util.cos(D), h = m = k * a.util.sin(D);
        return { tl: { x: i - m, y: r - p }, tr: { x: i + u, y: r - h }, bl: { x: i - u, y: r + h }, br: { x: i + m, y: r + p } };
      }, render: function (c, f, i, r, s) {
        ((r = r || {}).cornerStyle || s.cornerStyle) === "circle" ? a.controlsUtils.renderCircleControl.call(this, c, f, i, r, s) : a.controlsUtils.renderSquareControl.call(this, c, f, i, r, s);
      }
    };
  }(t), function () {
    function l(c, f) {
      var i, r, s, u, h = c.getAttribute("style"), p = c.getAttribute("offset") || 0;
      if (p = (p = parseFloat(p) / (/%$/.test(p) ? 100 : 1)) < 0 ? 0 : p > 1 ? 1 : p, h) {
        var m = h.split(/\s*;\s*/);
        for (m[m.length - 1] === "" && m.pop(), u = m.length; u--;) {
          var y = m[u].split(/\s*:\s*/), x = y[0].trim(), S = y[1].trim();
          x === "stop-color" ? i = S : x === "stop-opacity" && (s = S);
        }
      }
      return i || (i = c.getAttribute("stop-color") || "rgb(0,0,0)"), s || (s = c.getAttribute("stop-opacity")), r = (i = new v.Color(i)).getAlpha(), s = isNaN(parseFloat(s)) ? 1 : parseFloat(s), s *= r * f, { offset: p, color: i.toRgb(), opacity: s };
    }
    var a = v.util.object.clone;
    v.Gradient = v.util.createClass({
      offsetX: 0, offsetY: 0, gradientTransform: null, gradientUnits: "pixels", type: "linear", initialize: function (c) {
        c || (c = {}), c.coords || (c.coords = {});
        var f, i = this;
        Object.keys(c).forEach(function (r) {
          i[r] = c[r];
        }), this.id ? this.id += "_" + v.Object.__uid++ : this.id = v.Object.__uid++, f = { x1: c.coords.x1 || 0, y1: c.coords.y1 || 0, x2: c.coords.x2 || 0, y2: c.coords.y2 || 0 }, this.type === "radial" && (f.r1 = c.coords.r1 || 0, f.r2 = c.coords.r2 || 0), this.coords = f, this.colorStops = c.colorStops.slice();
      }, addColorStop: function (c) {
        for (var f in c) {
          var i = new v.Color(c[f]);
          this.colorStops.push({ offset: parseFloat(f), color: i.toRgb(), opacity: i.getAlpha() });
        }
        return this;
      }, toObject: function (c) {
        var f = { type: this.type, coords: this.coords, colorStops: this.colorStops, offsetX: this.offsetX, offsetY: this.offsetY, gradientUnits: this.gradientUnits, gradientTransform: this.gradientTransform ? this.gradientTransform.concat() : this.gradientTransform };
        return v.util.populateWithProperties(this, f, c), f;
      }, toSVG: function (c, f) {
        var i, r, s, u, h = a(this.coords, !0), p = (f = f || {}, a(this.colorStops, !0)), m = h.r1 > h.r2, y = this.gradientTransform ? this.gradientTransform.concat() : v.iMatrix.concat(), x = -this.offsetX, S = -this.offsetY, k = !!f.additionalTransform, D = this.gradientUnits === "pixels" ? "userSpaceOnUse" : "objectBoundingBox";
        if (p.sort(function (N, W) {
          return N.offset - W.offset;
        }), D === "objectBoundingBox" ? (x /= c.width, S /= c.height) : (x += c.width / 2, S += c.height / 2), c.type === "path" && this.gradientUnits !== "percentage" && (x -= c.pathOffset.x, S -= c.pathOffset.y), y[4] -= x, y[5] -= S, u = 'id="SVGID_' + this.id + '" gradientUnits="' + D + '"', u += ' gradientTransform="' + (k ? f.additionalTransform + " " : "") + v.util.matrixToSVG(y) + '" ', this.type === "linear" ? s = ["<linearGradient ", u, ' x1="', h.x1, '" y1="', h.y1, '" x2="', h.x2, '" y2="', h.y2, `">
`] : this.type === "radial" && (s = ["<radialGradient ", u, ' cx="', m ? h.x1 : h.x2, '" cy="', m ? h.y1 : h.y2, '" r="', m ? h.r1 : h.r2, '" fx="', m ? h.x2 : h.x1, '" fy="', m ? h.y2 : h.y1, `">
`]), this.type === "radial") {
          if (m) for ((p = p.concat()).reverse(), i = 0, r = p.length; i < r; i++) p[i].offset = 1 - p[i].offset;
          var M = Math.min(h.r1, h.r2);
          if (M > 0) {
            var C = M / Math.max(h.r1, h.r2);
            for (i = 0, r = p.length; i < r; i++) p[i].offset += C * (1 - p[i].offset);
          }
        }
        for (i = 0, r = p.length; i < r; i++) {
          var O = p[i];
          s.push("<stop ", 'offset="', 100 * O.offset + "%", '" style="stop-color:', O.color, O.opacity !== void 0 ? ";stop-opacity: " + O.opacity : ";", `"/>
`);
        }
        return s.push(this.type === "linear" ? `</linearGradient>
` : `</radialGradient>
`), s.join("");
      }, toLive: function (c) {
        var f, i, r, s = v.util.object.clone(this.coords);
        if (this.type) {
          for (this.type === "linear" ? f = c.createLinearGradient(s.x1, s.y1, s.x2, s.y2) : this.type === "radial" && (f = c.createRadialGradient(s.x1, s.y1, s.r1, s.x2, s.y2, s.r2)), i = 0, r = this.colorStops.length; i < r; i++) {
            var u = this.colorStops[i].color, h = this.colorStops[i].opacity, p = this.colorStops[i].offset;
            h !== void 0 && (u = new v.Color(u).setAlpha(h).toRgba()), f.addColorStop(p, u);
          }
          return f;
        }
      }
    }), v.util.object.extend(v.Gradient, {
      fromElement: function (c, f, i, r) {
        var s = parseFloat(i) / (/%$/.test(i) ? 100 : 1);
        s = s < 0 ? 0 : s > 1 ? 1 : s, isNaN(s) && (s = 1);
        var u, h, p, m, y = c.getElementsByTagName("stop"), x = c.getAttribute("gradientUnits") === "userSpaceOnUse" ? "pixels" : "percentage", S = c.getAttribute("gradientTransform") || "", k = [], D = 0, M = 0;
        for (c.nodeName === "linearGradient" || c.nodeName === "LINEARGRADIENT" ? (u = "linear", h = function (C) {
          return { x1: C.getAttribute("x1") || 0, y1: C.getAttribute("y1") || 0, x2: C.getAttribute("x2") || "100%", y2: C.getAttribute("y2") || 0 };
        }(c)) : (u = "radial", h = function (C) {
          return { x1: C.getAttribute("fx") || C.getAttribute("cx") || "50%", y1: C.getAttribute("fy") || C.getAttribute("cy") || "50%", r1: 0, x2: C.getAttribute("cx") || "50%", y2: C.getAttribute("cy") || "50%", r2: C.getAttribute("r") || "50%" };
        }(c)), p = y.length; p--;) k.push(l(y[p], s));
        return m = v.parseTransformAttribute(S), function (C, O, N, W) {
          var X, I;
          Object.keys(O).forEach(function (T) {
            (X = O[T]) === "Infinity" ? I = 1 : X === "-Infinity" ? I = 0 : (I = parseFloat(O[T], 10), typeof X == "string" && /^(\d+\.\d+)%|(\d+)%$/.test(X) && (I *= 0.01, W === "pixels" && (T !== "x1" && T !== "x2" && T !== "r2" || (I *= N.viewBoxWidth || N.width), T !== "y1" && T !== "y2" || (I *= N.viewBoxHeight || N.height)))), O[T] = I;
          });
        }(0, h, r, x), x === "pixels" && (D = -f.left, M = -f.top), new v.Gradient({ id: c.getAttribute("id"), type: u, coords: h, colorStops: k, gradientUnits: x, gradientTransform: m, offsetX: D, offsetY: M });
      }
    });
  }(), L = v.util.toFixed, v.Pattern = v.util.createClass({
    repeat: "repeat", offsetX: 0, offsetY: 0, crossOrigin: "", patternTransform: null, initialize: function (l, a) {
      if (l || (l = {}), this.id = v.Object.__uid++, this.setOptions(l), !l.source || l.source && typeof l.source != "string") a && a(this);
      else {
        var c = this;
        this.source = v.util.createImage(), v.util.loadImage(l.source, function (f, i) {
          c.source = f, a && a(c, i);
        }, null, this.crossOrigin);
      }
    }, toObject: function (l) {
      var a, c, f = v.Object.NUM_FRACTION_DIGITS;
      return typeof this.source.src == "string" ? a = this.source.src : typeof this.source == "object" && this.source.toDataURL && (a = this.source.toDataURL()), c = { type: "pattern", source: a, repeat: this.repeat, crossOrigin: this.crossOrigin, offsetX: L(this.offsetX, f), offsetY: L(this.offsetY, f), patternTransform: this.patternTransform ? this.patternTransform.concat() : null }, v.util.populateWithProperties(this, c, l), c;
    }, toSVG: function (l) {
      var a = typeof this.source == "function" ? this.source() : this.source, c = a.width / l.width, f = a.height / l.height, i = this.offsetX / l.width, r = this.offsetY / l.height, s = "";
      return this.repeat !== "repeat-x" && this.repeat !== "no-repeat" || (f = 1, r && (f += Math.abs(r))), this.repeat !== "repeat-y" && this.repeat !== "no-repeat" || (c = 1, i && (c += Math.abs(i))), a.src ? s = a.src : a.toDataURL && (s = a.toDataURL()), '<pattern id="SVGID_' + this.id + '" x="' + i + '" y="' + r + '" width="' + c + '" height="' + f + `">
<image x="0" y="0" width="` + a.width + '" height="' + a.height + '" xlink:href="' + s + `"></image>
</pattern>
`;
    }, setOptions: function (l) {
      for (var a in l) this[a] = l[a];
    }, toLive: function (l) {
      var a = this.source;
      return !a || a.src !== void 0 && (!a.complete || a.naturalWidth === 0 || a.naturalHeight === 0) ? "" : l.createPattern(a, this.repeat);
    }
  }), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.toFixed;
    a.Shadow ? a.warn("fabric.Shadow is already defined.") : (a.Shadow = a.util.createClass({
      color: "rgb(0,0,0)", blur: 0, offsetX: 0, offsetY: 0, affectStroke: !1, includeDefaultValues: !0, nonScaling: !1, initialize: function (f) {
        for (var i in typeof f == "string" && (f = this._parseShadow(f)), f) this[i] = f[i];
        this.id = a.Object.__uid++;
      }, _parseShadow: function (f) {
        var i = f.trim(), r = a.Shadow.reOffsetsAndBlur.exec(i) || [];
        return { color: (i.replace(a.Shadow.reOffsetsAndBlur, "") || "rgb(0,0,0)").trim(), offsetX: parseFloat(r[1], 10) || 0, offsetY: parseFloat(r[2], 10) || 0, blur: parseFloat(r[3], 10) || 0 };
      }, toString: function () {
        return [this.offsetX, this.offsetY, this.blur, this.color].join("px ");
      }, toSVG: function (f) {
        var i = 40, r = 40, s = a.Object.NUM_FRACTION_DIGITS, u = a.util.rotateVector({ x: this.offsetX, y: this.offsetY }, a.util.degreesToRadians(-f.angle)), h = new a.Color(this.color);
        return f.width && f.height && (i = 100 * c((Math.abs(u.x) + this.blur) / f.width, s) + 20, r = 100 * c((Math.abs(u.y) + this.blur) / f.height, s) + 20), f.flipX && (u.x *= -1), f.flipY && (u.y *= -1), '<filter id="SVGID_' + this.id + '" y="-' + r + '%" height="' + (100 + 2 * r) + '%" x="-' + i + '%" width="' + (100 + 2 * i) + `%" >
    <feGaussianBlur in="SourceAlpha" stdDeviation="` + c(this.blur ? this.blur / 2 : 0, s) + `"></feGaussianBlur>
    <feOffset dx="` + c(u.x, s) + '" dy="' + c(u.y, s) + `" result="oBlur" ></feOffset>
    <feFlood flood-color="` + h.toRgb() + '" flood-opacity="' + h.getAlpha() + `"/>
    <feComposite in2="oBlur" operator="in" />
    <feMerge>
        <feMergeNode></feMergeNode>
        <feMergeNode in="SourceGraphic"></feMergeNode>
    </feMerge>
</filter>
`;
      }, toObject: function () {
        if (this.includeDefaultValues) return { color: this.color, blur: this.blur, offsetX: this.offsetX, offsetY: this.offsetY, affectStroke: this.affectStroke, nonScaling: this.nonScaling };
        var f = {}, i = a.Shadow.prototype;
        return ["color", "blur", "offsetX", "offsetY", "affectStroke", "nonScaling"].forEach(function (r) {
          this[r] !== i[r] && (f[r] = this[r]);
        }, this), f;
      }
    }), a.Shadow.reOffsetsAndBlur = /(?:\s|^)(-?\d+(?:\.\d*)?(?:px)?(?:\s?|$))?(-?\d+(?:\.\d*)?(?:px)?(?:\s?|$))?(\d+(?:\.\d*)?(?:px)?)?(?:\s?|$)(?:$|\s)/);
  }(t), function () {
    if (v.StaticCanvas) v.warn("fabric.StaticCanvas is already defined.");
    else {
      var l = v.util.object.extend, a = v.util.getElementOffset, c = v.util.removeFromArray, f = v.util.toFixed, i = v.util.transformPoint, r = v.util.invertTransform, s = v.util.getNodeCanvas, u = v.util.createCanvasElement, h = new Error("Could not initialize `canvas` element");
      v.StaticCanvas = v.util.createClass(v.CommonMethods, {
        initialize: function (p, m) {
          m || (m = {}), this.renderAndResetBound = this.renderAndReset.bind(this), this.requestRenderAllBound = this.requestRenderAll.bind(this), this._initStatic(p, m);
        }, backgroundColor: "", backgroundImage: null, overlayColor: "", overlayImage: null, includeDefaultValues: !0, stateful: !1, renderOnAddRemove: !0, controlsAboveOverlay: !1, allowTouchScrolling: !1, imageSmoothingEnabled: !0, viewportTransform: v.iMatrix.concat(), backgroundVpt: !0, overlayVpt: !0, enableRetinaScaling: !0, vptCoords: {}, skipOffscreen: !0, clipPath: void 0, _initStatic: function (p, m) {
          var y = this.requestRenderAllBound;
          this._objects = [], this._createLowerCanvas(p), this._initOptions(m), this.interactive || this._initRetinaScaling(), m.overlayImage && this.setOverlayImage(m.overlayImage, y), m.backgroundImage && this.setBackgroundImage(m.backgroundImage, y), m.backgroundColor && this.setBackgroundColor(m.backgroundColor, y), m.overlayColor && this.setOverlayColor(m.overlayColor, y), this.calcOffset();
        }, _isRetinaScaling: function () {
          return v.devicePixelRatio !== 1 && this.enableRetinaScaling;
        }, getRetinaScaling: function () {
          return this._isRetinaScaling() ? v.devicePixelRatio : 1;
        }, _initRetinaScaling: function () {
          if (this._isRetinaScaling()) {
            var p = v.devicePixelRatio;
            this.__initRetinaScaling(p, this.lowerCanvasEl, this.contextContainer), this.upperCanvasEl && this.__initRetinaScaling(p, this.upperCanvasEl, this.contextTop);
          }
        }, __initRetinaScaling: function (p, m, y) {
          m.setAttribute("width", this.width * p), m.setAttribute("height", this.height * p), y.scale(p, p);
        }, calcOffset: function () {
          return this._offset = a(this.lowerCanvasEl), this;
        }, setOverlayImage: function (p, m, y) {
          return this.__setBgOverlayImage("overlayImage", p, m, y);
        }, setBackgroundImage: function (p, m, y) {
          return this.__setBgOverlayImage("backgroundImage", p, m, y);
        }, setOverlayColor: function (p, m) {
          return this.__setBgOverlayColor("overlayColor", p, m);
        }, setBackgroundColor: function (p, m) {
          return this.__setBgOverlayColor("backgroundColor", p, m);
        }, __setBgOverlayImage: function (p, m, y, x) {
          return typeof m == "string" ? v.util.loadImage(m, function (S, k) {
            if (S) {
              var D = new v.Image(S, x);
              this[p] = D, D.canvas = this;
            }
            y && y(S, k);
          }, this, x && x.crossOrigin) : (x && m.setOptions(x), this[p] = m, m && (m.canvas = this), y && y(m, !1)), this;
        }, __setBgOverlayColor: function (p, m, y) {
          return this[p] = m, this._initGradient(m, p), this._initPattern(m, p, y), this;
        }, _createCanvasElement: function () {
          var p = u();
          if (!p || (p.style || (p.style = {}), p.getContext === void 0)) throw h;
          return p;
        }, _initOptions: function (p) {
          var m = this.lowerCanvasEl;
          this._setOptions(p), this.width = this.width || parseInt(m.width, 10) || 0, this.height = this.height || parseInt(m.height, 10) || 0, this.lowerCanvasEl.style && (m.width = this.width, m.height = this.height, m.style.width = this.width + "px", m.style.height = this.height + "px", this.viewportTransform = this.viewportTransform.slice());
        }, _createLowerCanvas: function (p) {
          p && p.getContext ? this.lowerCanvasEl = p : this.lowerCanvasEl = v.util.getById(p) || this._createCanvasElement(), v.util.addClass(this.lowerCanvasEl, "lower-canvas"), this._originalCanvasStyle = this.lowerCanvasEl.style, this.interactive && this._applyCanvasStyle(this.lowerCanvasEl), this.contextContainer = this.lowerCanvasEl.getContext("2d");
        }, getWidth: function () {
          return this.width;
        }, getHeight: function () {
          return this.height;
        }, setWidth: function (p, m) {
          return this.setDimensions({ width: p }, m);
        }, setHeight: function (p, m) {
          return this.setDimensions({ height: p }, m);
        }, setDimensions: function (p, m) {
          var y;
          for (var x in m = m || {}, p) y = p[x], m.cssOnly || (this._setBackstoreDimension(x, p[x]), y += "px", this.hasLostContext = !0), m.backstoreOnly || this._setCssDimension(x, y);
          return this._isCurrentlyDrawing && this.freeDrawingBrush && this.freeDrawingBrush._setBrushStyles(), this._initRetinaScaling(), this.calcOffset(), m.cssOnly || this.requestRenderAll(), this;
        }, _setBackstoreDimension: function (p, m) {
          return this.lowerCanvasEl[p] = m, this.upperCanvasEl && (this.upperCanvasEl[p] = m), this.cacheCanvasEl && (this.cacheCanvasEl[p] = m), this[p] = m, this;
        }, _setCssDimension: function (p, m) {
          return this.lowerCanvasEl.style[p] = m, this.upperCanvasEl && (this.upperCanvasEl.style[p] = m), this.wrapperEl && (this.wrapperEl.style[p] = m), this;
        }, getZoom: function () {
          return this.viewportTransform[0];
        }, setViewportTransform: function (p) {
          var m, y, x, S = this._activeObject, k = this.backgroundImage, D = this.overlayImage;
          for (this.viewportTransform = p, y = 0, x = this._objects.length; y < x; y++) (m = this._objects[y]).group || m.setCoords(!0);
          return S && S.setCoords(), k && k.setCoords(!0), D && D.setCoords(!0), this.calcViewportBoundaries(), this.renderOnAddRemove && this.requestRenderAll(), this;
        }, zoomToPoint: function (p, m) {
          var y = p, x = this.viewportTransform.slice(0);
          p = i(p, r(this.viewportTransform)), x[0] = m, x[3] = m;
          var S = i(p, x);
          return x[4] += y.x - S.x, x[5] += y.y - S.y, this.setViewportTransform(x);
        }, setZoom: function (p) {
          return this.zoomToPoint(new v.Point(0, 0), p), this;
        }, absolutePan: function (p) {
          var m = this.viewportTransform.slice(0);
          return m[4] = -p.x, m[5] = -p.y, this.setViewportTransform(m);
        }, relativePan: function (p) {
          return this.absolutePan(new v.Point(-p.x - this.viewportTransform[4], -p.y - this.viewportTransform[5]));
        }, getElement: function () {
          return this.lowerCanvasEl;
        }, _onObjectAdded: function (p) {
          this.stateful && p.setupState(), p._set("canvas", this), p.setCoords(), this.fire("object:added", { target: p }), p.fire("added");
        }, _onObjectRemoved: function (p) {
          this.fire("object:removed", { target: p }), p.fire("removed"), delete p.canvas;
        }, clearContext: function (p) {
          return p.clearRect(0, 0, this.width, this.height), this;
        }, getContext: function () {
          return this.contextContainer;
        }, clear: function () {
          return this.remove.apply(this, this.getObjects()), this.backgroundImage = null, this.overlayImage = null, this.backgroundColor = "", this.overlayColor = "", this._hasITextHandlers && (this.off("mouse:up", this._mouseUpITextHandler), this._iTextInstances = null, this._hasITextHandlers = !1), this.clearContext(this.contextContainer), this.fire("canvas:cleared"), this.renderOnAddRemove && this.requestRenderAll(), this;
        }, renderAll: function () {
          var p = this.contextContainer;
          return this.renderCanvas(p, this._objects), this;
        }, renderAndReset: function () {
          this.isRendering = 0, this.renderAll();
        }, requestRenderAll: function () {
          return this.isRendering || (this.isRendering = v.util.requestAnimFrame(this.renderAndResetBound)), this;
        }, calcViewportBoundaries: function () {
          var p = {}, m = this.width, y = this.height, x = r(this.viewportTransform);
          return p.tl = i({ x: 0, y: 0 }, x), p.br = i({ x: m, y }, x), p.tr = new v.Point(p.br.x, p.tl.y), p.bl = new v.Point(p.tl.x, p.br.y), this.vptCoords = p, p;
        }, cancelRequestedRender: function () {
          this.isRendering && (v.util.cancelAnimFrame(this.isRendering), this.isRendering = 0);
        }, renderCanvas: function (p, m) {
          var y = this.viewportTransform, x = this.clipPath;
          this.cancelRequestedRender(), this.calcViewportBoundaries(), this.clearContext(p), v.util.setImageSmoothing(p, this.imageSmoothingEnabled), this.fire("before:render", { ctx: p }), this._renderBackground(p), p.save(), p.transform(y[0], y[1], y[2], y[3], y[4], y[5]), this._renderObjects(p, m), p.restore(), !this.controlsAboveOverlay && this.interactive && this.drawControls(p), x && (x.canvas = this, x.shouldCache(), x._transformDone = !0, x.renderCache({ forClipping: !0 }), this.drawClipPathOnCanvas(p)), this._renderOverlay(p), this.controlsAboveOverlay && this.interactive && this.drawControls(p), this.fire("after:render", { ctx: p });
        }, drawClipPathOnCanvas: function (p) {
          var m = this.viewportTransform, y = this.clipPath;
          p.save(), p.transform(m[0], m[1], m[2], m[3], m[4], m[5]), p.globalCompositeOperation = "destination-in", y.transform(p), p.scale(1 / y.zoomX, 1 / y.zoomY), p.drawImage(y._cacheCanvas, -y.cacheTranslationX, -y.cacheTranslationY), p.restore();
        }, _renderObjects: function (p, m) {
          var y, x;
          for (y = 0, x = m.length; y < x; ++y) m[y] && m[y].render(p);
        }, _renderBackgroundOrOverlay: function (p, m) {
          var y = this[m + "Color"], x = this[m + "Image"], S = this.viewportTransform, k = this[m + "Vpt"];
          if (y || x) {
            if (y) {
              p.save(), p.beginPath(), p.moveTo(0, 0), p.lineTo(this.width, 0), p.lineTo(this.width, this.height), p.lineTo(0, this.height), p.closePath(), p.fillStyle = y.toLive ? y.toLive(p, this) : y, k && p.transform(S[0], S[1], S[2], S[3], S[4], S[5]), p.transform(1, 0, 0, 1, y.offsetX || 0, y.offsetY || 0);
              var D = y.gradientTransform || y.patternTransform;
              D && p.transform(D[0], D[1], D[2], D[3], D[4], D[5]), p.fill(), p.restore();
            }
            x && (p.save(), k && p.transform(S[0], S[1], S[2], S[3], S[4], S[5]), x.render(p), p.restore());
          }
        }, _renderBackground: function (p) {
          this._renderBackgroundOrOverlay(p, "background");
        }, _renderOverlay: function (p) {
          this._renderBackgroundOrOverlay(p, "overlay");
        }, getCenter: function () {
          return { top: this.height / 2, left: this.width / 2 };
        }, centerObjectH: function (p) {
          return this._centerObject(p, new v.Point(this.getCenter().left, p.getCenterPoint().y));
        }, centerObjectV: function (p) {
          return this._centerObject(p, new v.Point(p.getCenterPoint().x, this.getCenter().top));
        }, centerObject: function (p) {
          var m = this.getCenter();
          return this._centerObject(p, new v.Point(m.left, m.top));
        }, viewportCenterObject: function (p) {
          var m = this.getVpCenter();
          return this._centerObject(p, m);
        }, viewportCenterObjectH: function (p) {
          var m = this.getVpCenter();
          return this._centerObject(p, new v.Point(m.x, p.getCenterPoint().y)), this;
        }, viewportCenterObjectV: function (p) {
          var m = this.getVpCenter();
          return this._centerObject(p, new v.Point(p.getCenterPoint().x, m.y));
        }, getVpCenter: function () {
          var p = this.getCenter(), m = r(this.viewportTransform);
          return i({ x: p.left, y: p.top }, m);
        }, _centerObject: function (p, m) {
          return p.setPositionByOrigin(m, "center", "center"), p.setCoords(), this.renderOnAddRemove && this.requestRenderAll(), this;
        }, toDatalessJSON: function (p) {
          return this.toDatalessObject(p);
        }, toObject: function (p) {
          return this._toObjectMethod("toObject", p);
        }, toDatalessObject: function (p) {
          return this._toObjectMethod("toDatalessObject", p);
        }, _toObjectMethod: function (p, m) {
          var y = this.clipPath, x = { version: v.version, objects: this._toObjects(p, m) };
          return y && !y.excludeFromExport && (x.clipPath = this._toObject(this.clipPath, p, m)), l(x, this.__serializeBgOverlay(p, m)), v.util.populateWithProperties(this, x, m), x;
        }, _toObjects: function (p, m) {
          return this._objects.filter(function (y) {
            return !y.excludeFromExport;
          }).map(function (y) {
            return this._toObject(y, p, m);
          }, this);
        }, _toObject: function (p, m, y) {
          var x;
          this.includeDefaultValues || (x = p.includeDefaultValues, p.includeDefaultValues = !1);
          var S = p[m](y);
          return this.includeDefaultValues || (p.includeDefaultValues = x), S;
        }, __serializeBgOverlay: function (p, m) {
          var y = {}, x = this.backgroundImage, S = this.overlayImage, k = this.backgroundColor, D = this.overlayColor;
          return k && k.toObject ? k.excludeFromExport || (y.background = k.toObject(m)) : k && (y.background = k), D && D.toObject ? D.excludeFromExport || (y.overlay = D.toObject(m)) : D && (y.overlay = D), x && !x.excludeFromExport && (y.backgroundImage = this._toObject(x, p, m)), S && !S.excludeFromExport && (y.overlayImage = this._toObject(S, p, m)), y;
        }, svgViewportTransformation: !0, toSVG: function (p, m) {
          p || (p = {}), p.reviver = m;
          var y = [];
          return this._setSVGPreamble(y, p), this._setSVGHeader(y, p), this.clipPath && y.push('<g clip-path="url(#' + this.clipPath.clipPathId + `)" >
`), this._setSVGBgOverlayColor(y, "background"), this._setSVGBgOverlayImage(y, "backgroundImage", m), this._setSVGObjects(y, m), this.clipPath && y.push(`</g>
`), this._setSVGBgOverlayColor(y, "overlay"), this._setSVGBgOverlayImage(y, "overlayImage", m), y.push("</svg>"), y.join("");
        }, _setSVGPreamble: function (p, m) {
          m.suppressPreamble || p.push('<?xml version="1.0" encoding="', m.encoding || "UTF-8", `" standalone="no" ?>
`, '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" ', `"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
`);
        }, _setSVGHeader: function (p, m) {
          var y, x = m.width || this.width, S = m.height || this.height, k = 'viewBox="0 0 ' + this.width + " " + this.height + '" ', D = v.Object.NUM_FRACTION_DIGITS;
          m.viewBox ? k = 'viewBox="' + m.viewBox.x + " " + m.viewBox.y + " " + m.viewBox.width + " " + m.viewBox.height + '" ' : this.svgViewportTransformation && (y = this.viewportTransform, k = 'viewBox="' + f(-y[4] / y[0], D) + " " + f(-y[5] / y[3], D) + " " + f(this.width / y[0], D) + " " + f(this.height / y[3], D) + '" '), p.push("<svg ", 'xmlns="http://www.w3.org/2000/svg" ', 'xmlns:xlink="http://www.w3.org/1999/xlink" ', 'version="1.1" ', 'width="', x, '" ', 'height="', S, '" ', k, `xml:space="preserve">
`, "<desc>Created with Fabric.js ", v.version, `</desc>
`, `<defs>
`, this.createSVGFontFacesMarkup(), this.createSVGRefElementsMarkup(), this.createSVGClipPathMarkup(m), `</defs>
`);
        }, createSVGClipPathMarkup: function (p) {
          var m = this.clipPath;
          return m ? (m.clipPathId = "CLIPPATH_" + v.Object.__uid++, '<clipPath id="' + m.clipPathId + `" >
` + this.clipPath.toClipPathSVG(p.reviver) + `</clipPath>
`) : "";
        }, createSVGRefElementsMarkup: function () {
          var p = this;
          return ["background", "overlay"].map(function (m) {
            var y = p[m + "Color"];
            if (y && y.toLive) {
              var x = p[m + "Vpt"], S = p.viewportTransform, k = { width: p.width / (x ? S[0] : 1), height: p.height / (x ? S[3] : 1) };
              return y.toSVG(k, { additionalTransform: x ? v.util.matrixToSVG(S) : "" });
            }
          }).join("");
        }, createSVGFontFacesMarkup: function () {
          var p, m, y, x, S, k, D, M, C = "", O = {}, N = v.fontPaths, W = [];
          for (this._objects.forEach(function I(T) {
            W.push(T), T._objects && T._objects.forEach(I);
          }), D = 0, M = W.length; D < M; D++) if (m = (p = W[D]).fontFamily, p.type.indexOf("text") !== -1 && !O[m] && N[m] && (O[m] = !0, p.styles)) for (S in y = p.styles) for (k in x = y[S]) !O[m = x[k].fontFamily] && N[m] && (O[m] = !0);
          for (var X in O) C += [`        @font-face {
`, "            font-family: '", X, `';
`, "            src: url('", N[X], `');
`, `        }
`].join("");
          return C && (C = ['    <style type="text/css">', `<![CDATA[
`, C, "]]>", `</style>
`].join("")), C;
        }, _setSVGObjects: function (p, m) {
          var y, x, S, k = this._objects;
          for (x = 0, S = k.length; x < S; x++) (y = k[x]).excludeFromExport || this._setSVGObject(p, y, m);
        }, _setSVGObject: function (p, m, y) {
          p.push(m.toSVG(y));
        }, _setSVGBgOverlayImage: function (p, m, y) {
          this[m] && !this[m].excludeFromExport && this[m].toSVG && p.push(this[m].toSVG(y));
        }, _setSVGBgOverlayColor: function (p, m) {
          var y = this[m + "Color"], x = this.viewportTransform, S = this.width, k = this.height;
          if (y) if (y.toLive) {
            var D = y.repeat, M = v.util.invertTransform(x), C = this[m + "Vpt"] ? v.util.matrixToSVG(M) : "";
            p.push('<rect transform="' + C + " translate(", S / 2, ",", k / 2, ')"', ' x="', y.offsetX - S / 2, '" y="', y.offsetY - k / 2, '" ', 'width="', D === "repeat-y" || D === "no-repeat" ? y.source.width : S, '" height="', D === "repeat-x" || D === "no-repeat" ? y.source.height : k, '" fill="url(#SVGID_' + y.id + ')"', `></rect>
`);
          } else p.push('<rect x="0" y="0" width="100%" height="100%" ', 'fill="', y, '"', `></rect>
`);
        }, sendToBack: function (p) {
          if (!p) return this;
          var m, y, x, S = this._activeObject;
          if (p === S && p.type === "activeSelection") for (m = (x = S._objects).length; m--;) y = x[m], c(this._objects, y), this._objects.unshift(y);
          else c(this._objects, p), this._objects.unshift(p);
          return this.renderOnAddRemove && this.requestRenderAll(), this;
        }, bringToFront: function (p) {
          if (!p) return this;
          var m, y, x, S = this._activeObject;
          if (p === S && p.type === "activeSelection") for (x = S._objects, m = 0; m < x.length; m++) y = x[m], c(this._objects, y), this._objects.push(y);
          else c(this._objects, p), this._objects.push(p);
          return this.renderOnAddRemove && this.requestRenderAll(), this;
        }, sendBackwards: function (p, m) {
          if (!p) return this;
          var y, x, S, k, D, M = this._activeObject, C = 0;
          if (p === M && p.type === "activeSelection") for (D = M._objects, y = 0; y < D.length; y++) x = D[y], (S = this._objects.indexOf(x)) > 0 + C && (k = S - 1, c(this._objects, x), this._objects.splice(k, 0, x)), C++;
          else (S = this._objects.indexOf(p)) !== 0 && (k = this._findNewLowerIndex(p, S, m), c(this._objects, p), this._objects.splice(k, 0, p));
          return this.renderOnAddRemove && this.requestRenderAll(), this;
        }, _findNewLowerIndex: function (p, m, y) {
          var x, S;
          if (y) {
            for (x = m, S = m - 1; S >= 0; --S)
              if (p.intersectsWithObject(this._objects[S]) || p.isContainedWithinObject(this._objects[S]) || this._objects[S].isContainedWithinObject(p)) {
                x = S;
                break;
              }
          } else x = m - 1;
          return x;
        }, bringForward: function (p, m) {
          if (!p) return this;
          var y, x, S, k, D, M = this._activeObject, C = 0;
          if (p === M && p.type === "activeSelection") for (y = (D = M._objects).length; y--;) x = D[y], (S = this._objects.indexOf(x)) < this._objects.length - 1 - C && (k = S + 1, c(this._objects, x), this._objects.splice(k, 0, x)), C++;
          else (S = this._objects.indexOf(p)) !== this._objects.length - 1 && (k = this._findNewUpperIndex(p, S, m), c(this._objects, p), this._objects.splice(k, 0, p));
          return this.renderOnAddRemove && this.requestRenderAll(), this;
        }, _findNewUpperIndex: function (p, m, y) {
          var x, S, k;
          if (y) {
            for (x = m, S = m + 1, k = this._objects.length; S < k; ++S)
              if (p.intersectsWithObject(this._objects[S]) || p.isContainedWithinObject(this._objects[S]) || this._objects[S].isContainedWithinObject(p)) {
                x = S;
                break;
              }
          } else x = m + 1;
          return x;
        }, moveTo: function (p, m) {
          return c(this._objects, p), this._objects.splice(m, 0, p), this.renderOnAddRemove && this.requestRenderAll();
        }, dispose: function () {
          return this.isRendering && (v.util.cancelAnimFrame(this.isRendering), this.isRendering = 0), this.forEachObject(function (p) {
            p.dispose && p.dispose();
          }), this._objects = [], this.backgroundImage && this.backgroundImage.dispose && this.backgroundImage.dispose(), this.backgroundImage = null, this.overlayImage && this.overlayImage.dispose && this.overlayImage.dispose(), this.overlayImage = null, this._iTextInstances = null, this.contextContainer = null, this.lowerCanvasEl.classList.remove("lower-canvas"), this.lowerCanvasEl.style = this._originalCanvasStyle, delete this._originalCanvasStyle, this.lowerCanvasEl.setAttribute("width", this.width), this.lowerCanvasEl.setAttribute("height", this.height), v.util.cleanUpJsdomNode(this.lowerCanvasEl), this.lowerCanvasEl = void 0, this;
        }, toString: function () {
          return "#<fabric.Canvas (" + this.complexity() + "): { objects: " + this._objects.length + " }>";
        }
      }), l(v.StaticCanvas.prototype, v.Observable), l(v.StaticCanvas.prototype, v.Collection), l(v.StaticCanvas.prototype, v.DataURLExporter), l(v.StaticCanvas, {
        EMPTY_JSON: '{"objects": [], "background": "white"}', supports: function (p) {
          var m = u();
          if (!m || !m.getContext) return null;
          var y = m.getContext("2d");
          return y && p === "setLineDash" ? y.setLineDash !== void 0 : null;
        }
      }), v.StaticCanvas.prototype.toJSON = v.StaticCanvas.prototype.toObject, v.isLikelyNode && (v.StaticCanvas.prototype.createPNGStream = function () {
        var p = s(this.lowerCanvasEl);
        return p && p.createPNGStream();
      }, v.StaticCanvas.prototype.createJPEGStream = function (p) {
        var m = s(this.lowerCanvasEl);
        return m && m.createJPEGStream(p);
      });
    }
  }(), v.BaseBrush = v.util.createClass({
    color: "rgb(0, 0, 0)", width: 1, shadow: null, strokeLineCap: "round", strokeLineJoin: "round", strokeMiterLimit: 10, strokeDashArray: null, limitedToCanvasSize: !1, _setBrushStyles: function () {
      var l = this.canvas.contextTop;
      l.strokeStyle = this.color, l.lineWidth = this.width, l.lineCap = this.strokeLineCap, l.miterLimit = this.strokeMiterLimit, l.lineJoin = this.strokeLineJoin, l.setLineDash(this.strokeDashArray || []);
    }, _saveAndTransform: function (l) {
      var a = this.canvas.viewportTransform;
      l.save(), l.transform(a[0], a[1], a[2], a[3], a[4], a[5]);
    }, _setShadow: function () {
      if (this.shadow) {
        var l = this.canvas, a = this.shadow, c = l.contextTop, f = l.getZoom();
        l && l._isRetinaScaling() && (f *= v.devicePixelRatio), c.shadowColor = a.color, c.shadowBlur = a.blur * f, c.shadowOffsetX = a.offsetX * f, c.shadowOffsetY = a.offsetY * f;
      }
    }, needsFullRender: function () {
      return new v.Color(this.color).getAlpha() < 1 || !!this.shadow;
    }, _resetShadow: function () {
      var l = this.canvas.contextTop;
      l.shadowColor = "", l.shadowBlur = l.shadowOffsetX = l.shadowOffsetY = 0;
    }, _isOutSideCanvas: function (l) {
      return l.x < 0 || l.x > this.canvas.getWidth() || l.y < 0 || l.y > this.canvas.getHeight();
    }
  }), v.PencilBrush = v.util.createClass(v.BaseBrush, {
    decimate: 0.4, initialize: function (l) {
      this.canvas = l, this._points = [];
    }, _drawSegment: function (l, a, c) {
      var f = a.midPointFrom(c);
      return l.quadraticCurveTo(a.x, a.y, f.x, f.y), f;
    }, onMouseDown: function (l, a) {
      this.canvas._isMainEvent(a.e) && (this._prepareForDrawing(l), this._captureDrawingPath(l), this._render());
    }, onMouseMove: function (l, a) {
      if (this.canvas._isMainEvent(a.e) && (this.limitedToCanvasSize !== !0 || !this._isOutSideCanvas(l)) && this._captureDrawingPath(l) && this._points.length > 1) if (this.needsFullRender()) this.canvas.clearContext(this.canvas.contextTop), this._render();
      else {
        var c = this._points, f = c.length, i = this.canvas.contextTop;
        this._saveAndTransform(i), this.oldEnd && (i.beginPath(), i.moveTo(this.oldEnd.x, this.oldEnd.y)), this.oldEnd = this._drawSegment(i, c[f - 2], c[f - 1], !0), i.stroke(), i.restore();
      }
    }, onMouseUp: function (l) {
      return !this.canvas._isMainEvent(l.e) || (this.oldEnd = void 0, this._finalizeAndAddPath(), !1);
    }, _prepareForDrawing: function (l) {
      var a = new v.Point(l.x, l.y);
      this._reset(), this._addPoint(a), this.canvas.contextTop.moveTo(a.x, a.y);
    }, _addPoint: function (l) {
      return !(this._points.length > 1 && l.eq(this._points[this._points.length - 1]) || (this._points.push(l), 0));
    }, _reset: function () {
      this._points = [], this._setBrushStyles(), this._setShadow();
    }, _captureDrawingPath: function (l) {
      var a = new v.Point(l.x, l.y);
      return this._addPoint(a);
    }, _render: function () {
      var l, a, c = this.canvas.contextTop, f = this._points[0], i = this._points[1];
      if (this._saveAndTransform(c), c.beginPath(), this._points.length === 2 && f.x === i.x && f.y === i.y) {
        var r = this.width / 1e3;
        f = new v.Point(f.x, f.y), i = new v.Point(i.x, i.y), f.x -= r, i.x += r;
      }
      for (c.moveTo(f.x, f.y), l = 1, a = this._points.length; l < a; l++) this._drawSegment(c, f, i), f = this._points[l], i = this._points[l + 1];
      c.lineTo(f.x, f.y), c.stroke(), c.restore();
    }, convertPointsToSVGPath: function (l) {
      var a = this.width / 1e3;
      return v.util.getSmoothPathFromPoints(l, a);
    }, _isEmptySVGPath: function (l) {
      return v.util.joinPath(l) === "M 0 0 Q 0 0 0 0 L 0 0";
    }, createPath: function (l) {
      var a = new v.Path(l, { fill: null, stroke: this.color, strokeWidth: this.width, strokeLineCap: this.strokeLineCap, strokeMiterLimit: this.strokeMiterLimit, strokeLineJoin: this.strokeLineJoin, strokeDashArray: this.strokeDashArray });
      return this.shadow && (this.shadow.affectStroke = !0, a.shadow = new v.Shadow(this.shadow)), a;
    }, decimatePoints: function (l, a) {
      if (l.length <= 2) return l;
      var c, f = this.canvas.getZoom(), i = Math.pow(a / f, 2), r = l.length - 1, s = l[0], u = [s];
      for (c = 1; c < r - 1; c++) Math.pow(s.x - l[c].x, 2) + Math.pow(s.y - l[c].y, 2) >= i && (s = l[c], u.push(s));
      return u.push(l[r]), u;
    }, _finalizeAndAddPath: function () {
      this.canvas.contextTop.closePath(), this.decimate && (this._points = this.decimatePoints(this._points, this.decimate));
      var l = this.convertPointsToSVGPath(this._points);
      if (this._isEmptySVGPath(l)) this.canvas.requestRenderAll();
      else {
        var a = this.createPath(l);
        this.canvas.clearContext(this.canvas.contextTop), this.canvas.fire("before:path:created", { path: a }), this.canvas.add(a), this.canvas.requestRenderAll(), a.setCoords(), this._resetShadow(), this.canvas.fire("path:created", { path: a });
      }
    }
  }), v.CircleBrush = v.util.createClass(v.BaseBrush, {
    width: 10, initialize: function (l) {
      this.canvas = l, this.points = [];
    }, drawDot: function (l) {
      var a = this.addPoint(l), c = this.canvas.contextTop;
      this._saveAndTransform(c), this.dot(c, a), c.restore();
    }, dot: function (l, a) {
      l.fillStyle = a.fill, l.beginPath(), l.arc(a.x, a.y, a.radius, 0, 2 * Math.PI, !1), l.closePath(), l.fill();
    }, onMouseDown: function (l) {
      this.points.length = 0, this.canvas.clearContext(this.canvas.contextTop), this._setShadow(), this.drawDot(l);
    }, _render: function () {
      var l, a, c = this.canvas.contextTop, f = this.points;
      for (this._saveAndTransform(c), l = 0, a = f.length; l < a; l++) this.dot(c, f[l]);
      c.restore();
    }, onMouseMove: function (l) {
      this.limitedToCanvasSize === !0 && this._isOutSideCanvas(l) || (this.needsFullRender() ? (this.canvas.clearContext(this.canvas.contextTop), this.addPoint(l), this._render()) : this.drawDot(l));
    }, onMouseUp: function () {
      var l, a, c = this.canvas.renderOnAddRemove;
      this.canvas.renderOnAddRemove = !1;
      var f = [];
      for (l = 0, a = this.points.length; l < a; l++) {
        var i = this.points[l], r = new v.Circle({ radius: i.radius, left: i.x, top: i.y, originX: "center", originY: "center", fill: i.fill });
        this.shadow && (r.shadow = new v.Shadow(this.shadow)), f.push(r);
      }
      var s = new v.Group(f);
      s.canvas = this.canvas, this.canvas.fire("before:path:created", { path: s }), this.canvas.add(s), this.canvas.fire("path:created", { path: s }), this.canvas.clearContext(this.canvas.contextTop), this._resetShadow(), this.canvas.renderOnAddRemove = c, this.canvas.requestRenderAll();
    }, addPoint: function (l) {
      var a = new v.Point(l.x, l.y), c = v.util.getRandomInt(Math.max(0, this.width - 20), this.width + 20) / 2, f = new v.Color(this.color).setAlpha(v.util.getRandomInt(0, 100) / 100).toRgba();
      return a.radius = c, a.fill = f, this.points.push(a), a;
    }
  }), v.SprayBrush = v.util.createClass(v.BaseBrush, {
    width: 10, density: 20, dotWidth: 1, dotWidthVariance: 1, randomOpacity: !1, optimizeOverlapping: !0, initialize: function (l) {
      this.canvas = l, this.sprayChunks = [];
    }, onMouseDown: function (l) {
      this.sprayChunks.length = 0, this.canvas.clearContext(this.canvas.contextTop), this._setShadow(), this.addSprayChunk(l), this.render(this.sprayChunkPoints);
    }, onMouseMove: function (l) {
      this.limitedToCanvasSize === !0 && this._isOutSideCanvas(l) || (this.addSprayChunk(l), this.render(this.sprayChunkPoints));
    }, onMouseUp: function () {
      var l = this.canvas.renderOnAddRemove;
      this.canvas.renderOnAddRemove = !1;
      for (var a = [], c = 0, f = this.sprayChunks.length; c < f; c++) for (var i = this.sprayChunks[c], r = 0, s = i.length; r < s; r++) {
        var u = new v.Rect({ width: i[r].width, height: i[r].width, left: i[r].x + 1, top: i[r].y + 1, originX: "center", originY: "center", fill: this.color });
        a.push(u);
      }
      this.optimizeOverlapping && (a = this._getOptimizedRects(a));
      var h = new v.Group(a);
      this.shadow && h.set("shadow", new v.Shadow(this.shadow)), this.canvas.fire("before:path:created", { path: h }), this.canvas.add(h), this.canvas.fire("path:created", { path: h }), this.canvas.clearContext(this.canvas.contextTop), this._resetShadow(), this.canvas.renderOnAddRemove = l, this.canvas.requestRenderAll();
    }, _getOptimizedRects: function (l) {
      var a, c, f, i = {};
      for (c = 0, f = l.length; c < f; c++) i[a = l[c].left + "" + l[c].top] || (i[a] = l[c]);
      var r = [];
      for (a in i) r.push(i[a]);
      return r;
    }, render: function (l) {
      var a, c, f = this.canvas.contextTop;
      for (f.fillStyle = this.color, this._saveAndTransform(f), a = 0, c = l.length; a < c; a++) {
        var i = l[a];
        i.opacity !== void 0 && (f.globalAlpha = i.opacity), f.fillRect(i.x, i.y, i.width, i.width);
      }
      f.restore();
    }, _render: function () {
      var l, a, c = this.canvas.contextTop;
      for (c.fillStyle = this.color, this._saveAndTransform(c), l = 0, a = this.sprayChunks.length; l < a; l++) this.render(this.sprayChunks[l]);
      c.restore();
    }, addSprayChunk: function (l) {
      this.sprayChunkPoints = [];
      var a, c, f, i, r = this.width / 2;
      for (i = 0; i < this.density; i++) {
        a = v.util.getRandomInt(l.x - r, l.x + r), c = v.util.getRandomInt(l.y - r, l.y + r), f = this.dotWidthVariance ? v.util.getRandomInt(Math.max(1, this.dotWidth - this.dotWidthVariance), this.dotWidth + this.dotWidthVariance) : this.dotWidth;
        var s = new v.Point(a, c);
        s.width = f, this.randomOpacity && (s.opacity = v.util.getRandomInt(0, 100) / 100), this.sprayChunkPoints.push(s);
      }
      this.sprayChunks.push(this.sprayChunkPoints);
    }
  }), v.PatternBrush = v.util.createClass(v.PencilBrush, {
    getPatternSrc: function () {
      var l = v.util.createCanvasElement(), a = l.getContext("2d");
      return l.width = l.height = 25, a.fillStyle = this.color, a.beginPath(), a.arc(10, 10, 10, 0, 2 * Math.PI, !1), a.closePath(), a.fill(), l;
    }, getPatternSrcFunction: function () {
      return String(this.getPatternSrc).replace("this.color", '"' + this.color + '"');
    }, getPattern: function () {
      return this.canvas.contextTop.createPattern(this.source || this.getPatternSrc(), "repeat");
    }, _setBrushStyles: function () {
      this.callSuper("_setBrushStyles"), this.canvas.contextTop.strokeStyle = this.getPattern();
    }, createPath: function (l) {
      var a = this.callSuper("createPath", l), c = a._getLeftTopCoords().scalarAdd(a.strokeWidth / 2);
      return a.stroke = new v.Pattern({ source: this.source || this.getPatternSrcFunction(), offsetX: -c.x, offsetY: -c.y }), a;
    }
  }), function () {
    var l = v.util.getPointer, a = v.util.degreesToRadians, c = v.util.isTouchEvent;
    for (var f in v.Canvas = v.util.createClass(v.StaticCanvas, {
      initialize: function (i, r) {
        r || (r = {}), this.renderAndResetBound = this.renderAndReset.bind(this), this.requestRenderAllBound = this.requestRenderAll.bind(this), this._initStatic(i, r), this._initInteractive(), this._createCacheCanvas();
      }, uniformScaling: !0, uniScaleKey: "shiftKey", centeredScaling: !1, centeredRotation: !1, centeredKey: "altKey", altActionKey: "shiftKey", interactive: !0, selection: !0, selectionKey: "shiftKey", altSelectionKey: null, selectionColor: "rgba(100, 100, 255, 0.3)", selectionDashArray: [], selectionBorderColor: "rgba(255, 255, 255, 0.3)", selectionLineWidth: 1, selectionFullyContained: !1, hoverCursor: "move", moveCursor: "move", defaultCursor: "default", freeDrawingCursor: "crosshair", rotationCursor: "crosshair", notAllowedCursor: "not-allowed", containerClass: "canvas-container", perPixelTargetFind: !1, targetFindTolerance: 0, skipTargetFind: !1, isDrawingMode: !1, preserveObjectStacking: !1, snapAngle: 0, snapThreshold: null, stopContextMenu: !1, fireRightClick: !1, fireMiddleClick: !1, targets: [], _hoveredTarget: null, _hoveredTargets: [], _initInteractive: function () {
        this._currentTransform = null, this._groupSelector = null, this._initWrapperElement(), this._createUpperCanvas(), this._initEventListeners(), this._initRetinaScaling(), this.freeDrawingBrush = v.PencilBrush && new v.PencilBrush(this), this.calcOffset();
      }, _chooseObjectsToRender: function () {
        var i, r, s, u = this.getActiveObjects();
        if (u.length > 0 && !this.preserveObjectStacking) {
          r = [], s = [];
          for (var h = 0, p = this._objects.length; h < p; h++) i = this._objects[h], u.indexOf(i) === -1 ? r.push(i) : s.push(i);
          u.length > 1 && (this._activeObject._objects = s), r.push.apply(r, s);
        } else r = this._objects;
        return r;
      }, renderAll: function () {
        !this.contextTopDirty || this._groupSelector || this.isDrawingMode || (this.clearContext(this.contextTop), this.contextTopDirty = !1), this.hasLostContext && this.renderTopLayer(this.contextTop);
        var i = this.contextContainer;
        return this.renderCanvas(i, this._chooseObjectsToRender()), this;
      }, renderTopLayer: function (i) {
        i.save(), this.isDrawingMode && this._isCurrentlyDrawing && (this.freeDrawingBrush && this.freeDrawingBrush._render(), this.contextTopDirty = !0), this.selection && this._groupSelector && (this._drawSelection(i), this.contextTopDirty = !0), i.restore();
      }, renderTop: function () {
        var i = this.contextTop;
        return this.clearContext(i), this.renderTopLayer(i), this.fire("after:render"), this;
      }, _normalizePointer: function (i, r) {
        var s = i.calcTransformMatrix(), u = v.util.invertTransform(s), h = this.restorePointerVpt(r);
        return v.util.transformPoint(h, u);
      }, isTargetTransparent: function (i, r, s) {
        if (i.shouldCache() && i._cacheCanvas && i !== this._activeObject) {
          var u = this._normalizePointer(i, { x: r, y: s }), h = Math.max(i.cacheTranslationX + u.x * i.zoomX, 0), p = Math.max(i.cacheTranslationY + u.y * i.zoomY, 0);
          return v.util.isTransparent(i._cacheContext, Math.round(h), Math.round(p), this.targetFindTolerance);
        }
        var m = this.contextCache, y = i.selectionBackgroundColor, x = this.viewportTransform;
        return i.selectionBackgroundColor = "", this.clearContext(m), m.save(), m.transform(x[0], x[1], x[2], x[3], x[4], x[5]), i.render(m), m.restore(), i.selectionBackgroundColor = y, v.util.isTransparent(m, r, s, this.targetFindTolerance);
      }, _isSelectionKeyPressed: function (i) {
        return Object.prototype.toString.call(this.selectionKey) === "[object Array]" ? !!this.selectionKey.find(function (r) {
          return i[r] === !0;
        }) : i[this.selectionKey];
      }, _shouldClearSelection: function (i, r) {
        var s = this.getActiveObjects(), u = this._activeObject;
        return !r || r && u && s.length > 1 && s.indexOf(r) === -1 && u !== r && !this._isSelectionKeyPressed(i) || r && !r.evented || r && !r.selectable && u && u !== r;
      }, _shouldCenterTransform: function (i, r, s) {
        var u;
        if (i) return r === "scale" || r === "scaleX" || r === "scaleY" || r === "resizing" ? u = this.centeredScaling || i.centeredScaling : r === "rotate" && (u = this.centeredRotation || i.centeredRotation), u ? !s : s;
      }, _getOriginFromCorner: function (i, r) {
        var s = { x: i.originX, y: i.originY };
        return r === "ml" || r === "tl" || r === "bl" ? s.x = "right" : r !== "mr" && r !== "tr" && r !== "br" || (s.x = "left"), r === "tl" || r === "mt" || r === "tr" ? s.y = "bottom" : r !== "bl" && r !== "mb" && r !== "br" || (s.y = "top"), s;
      }, _getActionFromCorner: function (i, r, s, u) {
        if (!r || !i) return "drag";
        var h = u.controls[r];
        return h.getActionName(s, h, u);
      }, _setupCurrentTransform: function (i, r, s) {
        if (r) {
          var u = this.getPointer(i), h = r.__corner, p = r.controls[h], m = s && h ? p.getActionHandler(i, r, p) : v.controlsUtils.dragHandler, y = this._getActionFromCorner(s, h, i, r), x = this._getOriginFromCorner(r, h), S = i[this.centeredKey], k = { target: r, action: y, actionHandler: m, corner: h, scaleX: r.scaleX, scaleY: r.scaleY, skewX: r.skewX, skewY: r.skewY, offsetX: u.x - r.left, offsetY: u.y - r.top, originX: x.x, originY: x.y, ex: u.x, ey: u.y, lastX: u.x, lastY: u.y, theta: a(r.angle), width: r.width * r.scaleX, shiftKey: i.shiftKey, altKey: S, original: v.util.saveObjectTransform(r) };
          this._shouldCenterTransform(r, y, S) && (k.originX = "center", k.originY = "center"), k.original.originX = x.x, k.original.originY = x.y, this._currentTransform = k, this._beforeTransform(i);
        }
      }, setCursor: function (i) {
        this.upperCanvasEl.style.cursor = i;
      }, _drawSelection: function (i) {
        var r = this._groupSelector, s = new v.Point(r.ex, r.ey), u = v.util.transformPoint(s, this.viewportTransform), h = new v.Point(r.ex + r.left, r.ey + r.top), p = v.util.transformPoint(h, this.viewportTransform), m = Math.min(u.x, p.x), y = Math.min(u.y, p.y), x = Math.max(u.x, p.x), S = Math.max(u.y, p.y), k = this.selectionLineWidth / 2;
        this.selectionColor && (i.fillStyle = this.selectionColor, i.fillRect(m, y, x - m, S - y)), this.selectionLineWidth && this.selectionBorderColor && (i.lineWidth = this.selectionLineWidth, i.strokeStyle = this.selectionBorderColor, m += k, y += k, x -= k, S -= k, v.Object.prototype._setLineDash.call(this, i, this.selectionDashArray), i.strokeRect(m, y, x - m, S - y));
      }, findTarget: function (i, r) {
        if (!this.skipTargetFind) {
          var s, u, h = this.getPointer(i, !0), p = this._activeObject, m = this.getActiveObjects(), y = c(i), x = m.length > 1 && !r || m.length === 1;
          if (this.targets = [], x && p._findTargetCorner(h, y) || m.length > 1 && !r && p === this._searchPossibleTargets([p], h)) return p;
          if (m.length === 1 && p === this._searchPossibleTargets([p], h)) {
            if (!this.preserveObjectStacking) return p;
            s = p, u = this.targets, this.targets = [];
          }
          var S = this._searchPossibleTargets(this._objects, h);
          return i[this.altSelectionKey] && S && s && S !== s && (S = s, this.targets = u), S;
        }
      }, _checkTarget: function (i, r, s) {
        if (r && r.visible && r.evented && r.containsPoint(i) && (!this.perPixelTargetFind && !r.perPixelTargetFind || r.isEditing || !this.isTargetTransparent(r, s.x, s.y)))
          return !0;
      }, _searchPossibleTargets: function (i, r) {
        for (var s, u, h = i.length; h--;) {
          var p = i[h], m = p.group ? this._normalizePointer(p.group, r) : r;
          if (this._checkTarget(m, p, r)) {
            (s = i[h]).subTargetCheck && s instanceof v.Group && (u = this._searchPossibleTargets(s._objects, r)) && this.targets.push(u);
            break;
          }
        }
        return s;
      }, restorePointerVpt: function (i) {
        return v.util.transformPoint(i, v.util.invertTransform(this.viewportTransform));
      }, getPointer: function (i, r) {
        if (this._absolutePointer && !r) return this._absolutePointer;
        if (this._pointer && r) return this._pointer;
        var s, u = l(i), h = this.upperCanvasEl, p = h.getBoundingClientRect(), m = p.width || 0, y = p.height || 0;
        m && y || ("top" in p && "bottom" in p && (y = Math.abs(p.top - p.bottom)), "right" in p && "left" in p && (m = Math.abs(p.right - p.left))), this.calcOffset(), u.x = u.x - this._offset.left, u.y = u.y - this._offset.top, r || (u = this.restorePointerVpt(u));
        var x = this.getRetinaScaling();
        return x !== 1 && (u.x /= x, u.y /= x), s = m === 0 || y === 0 ? { width: 1, height: 1 } : { width: h.width / m, height: h.height / y }, { x: u.x * s.width, y: u.y * s.height };
      }, _createUpperCanvas: function () {
        var i = this.lowerCanvasEl.className.replace(/\s*lower-canvas\s*/, ""), r = this.lowerCanvasEl, s = this.upperCanvasEl;
        s ? s.className = "" : (s = this._createCanvasElement(), this.upperCanvasEl = s), v.util.addClass(s, "upper-canvas " + i), this.wrapperEl.appendChild(s), this._copyCanvasStyle(r, s), this._applyCanvasStyle(s), this.contextTop = s.getContext("2d");
      }, _createCacheCanvas: function () {
        this.cacheCanvasEl = this._createCanvasElement(), this.cacheCanvasEl.setAttribute("width", this.width), this.cacheCanvasEl.setAttribute("height", this.height), this.contextCache = this.cacheCanvasEl.getContext("2d");
      }, _initWrapperElement: function () {
        this.wrapperEl = v.util.wrapElement(this.lowerCanvasEl, "div", { class: this.containerClass }), v.util.setStyle(this.wrapperEl, { width: this.width + "px", height: this.height + "px", position: "relative" }), v.util.makeElementUnselectable(this.wrapperEl);
      }, _applyCanvasStyle: function (i) {
        var r = this.width || i.width, s = this.height || i.height;
        v.util.setStyle(i, { position: "absolute", width: r + "px", height: s + "px", left: 0, top: 0, "touch-action": this.allowTouchScrolling ? "manipulation" : "none", "-ms-touch-action": this.allowTouchScrolling ? "manipulation" : "none" }), i.width = r, i.height = s, v.util.makeElementUnselectable(i);
      }, _copyCanvasStyle: function (i, r) {
        r.style.cssText = i.style.cssText;
      }, getSelectionContext: function () {
        return this.contextTop;
      }, getSelectionElement: function () {
        return this.upperCanvasEl;
      }, getActiveObject: function () {
        return this._activeObject;
      }, getActiveObjects: function () {
        var i = this._activeObject;
        return i ? i.type === "activeSelection" && i._objects ? i._objects.slice(0) : [i] : [];
      }, _onObjectRemoved: function (i) {
        i === this._activeObject && (this.fire("before:selection:cleared", { target: i }), this._discardActiveObject(), this.fire("selection:cleared", { target: i }), i.fire("deselected")), i === this._hoveredTarget && (this._hoveredTarget = null, this._hoveredTargets = []), this.callSuper("_onObjectRemoved", i);
      }, _fireSelectionEvents: function (i, r) {
        var s = !1, u = this.getActiveObjects(), h = [], p = [];
        i.forEach(function (m) {
          u.indexOf(m) === -1 && (s = !0, m.fire("deselected", { e: r, target: m }), p.push(m));
        }), u.forEach(function (m) {
          i.indexOf(m) === -1 && (s = !0, m.fire("selected", { e: r, target: m }), h.push(m));
        }), i.length > 0 && u.length > 0 ? s && this.fire("selection:updated", { e: r, selected: h, deselected: p, updated: h[0] || p[0], target: this._activeObject }) : u.length > 0 ? this.fire("selection:created", { e: r, selected: h, target: this._activeObject }) : i.length > 0 && this.fire("selection:cleared", { e: r, deselected: p });
      }, setActiveObject: function (i, r) {
        var s = this.getActiveObjects();
        return this._setActiveObject(i, r), this._fireSelectionEvents(s, r), this;
      }, _setActiveObject: function (i, r) {
        return this._activeObject !== i && !!this._discardActiveObject(r, i) && !i.onSelect({ e: r }) && (this._activeObject = i, !0);
      }, _discardActiveObject: function (i, r) {
        var s = this._activeObject;
        if (s) {
          if (s.onDeselect({ e: i, object: r })) return !1;
          this._activeObject = null;
        }
        return !0;
      }, discardActiveObject: function (i) {
        var r = this.getActiveObjects(), s = this.getActiveObject();
        return r.length && this.fire("before:selection:cleared", { target: s, e: i }), this._discardActiveObject(i), this._fireSelectionEvents(r, i), this;
      }, dispose: function () {
        var i = this.wrapperEl;
        return this.removeListeners(), i.removeChild(this.upperCanvasEl), i.removeChild(this.lowerCanvasEl), this.contextCache = null, this.contextTop = null, ["upperCanvasEl", "cacheCanvasEl"].forEach((function (r) {
          v.util.cleanUpJsdomNode(this[r]), this[r] = void 0;
        }).bind(this)), i.parentNode && i.parentNode.replaceChild(this.lowerCanvasEl, this.wrapperEl), delete this.wrapperEl, v.StaticCanvas.prototype.dispose.call(this), this;
      }, clear: function () {
        return this.discardActiveObject(), this.clearContext(this.contextTop), this.callSuper("clear");
      }, drawControls: function (i) {
        var r = this._activeObject;
        r && r._renderControls(i);
      }, _toObject: function (i, r, s) {
        var u = this._realizeGroupTransformOnObject(i), h = this.callSuper("_toObject", i, r, s);
        return this._unwindGroupTransformOnObject(i, u), h;
      }, _realizeGroupTransformOnObject: function (i) {
        if (i.group && i.group.type === "activeSelection" && this._activeObject === i.group) {
          var r = {};
          return ["angle", "flipX", "flipY", "left", "scaleX", "scaleY", "skewX", "skewY", "top"].forEach(function (s) {
            r[s] = i[s];
          }), v.util.addTransformToObject(i, this._activeObject.calcOwnMatrix()), r;
        }
        return null;
      }, _unwindGroupTransformOnObject: function (i, r) {
        r && i.set(r);
      }, _setSVGObject: function (i, r, s) {
        var u = this._realizeGroupTransformOnObject(r);
        this.callSuper("_setSVGObject", i, r, s), this._unwindGroupTransformOnObject(r, u);
      }, setViewportTransform: function (i) {
        this.renderOnAddRemove && this._activeObject && this._activeObject.isEditing && this._activeObject.clearContextTop(), v.StaticCanvas.prototype.setViewportTransform.call(this, i);
      }
    }), v.StaticCanvas) f !== "prototype" && (v.Canvas[f] = v.StaticCanvas[f]);
  }(), function () {
    var l = v.util.addListener, a = v.util.removeListener, c = { passive: !1 };
    function f(i, r) {
      return i.button && i.button === r - 1;
    }
    v.util.object.extend(v.Canvas.prototype, {
      mainTouchId: null, _initEventListeners: function () {
        this.removeListeners(), this._bindEvents(), this.addOrRemove(l, "add");
      }, _getEventPrefix: function () {
        return this.enablePointerEvents ? "pointer" : "mouse";
      }, addOrRemove: function (i, r) {
        var s = this.upperCanvasEl, u = this._getEventPrefix();
        i(v.window, "resize", this._onResize), i(s, u + "down", this._onMouseDown), i(s, u + "move", this._onMouseMove, c), i(s, u + "out", this._onMouseOut), i(s, u + "enter", this._onMouseEnter), i(s, "wheel", this._onMouseWheel), i(s, "contextmenu", this._onContextMenu), i(s, "dblclick", this._onDoubleClick), i(s, "dragover", this._onDragOver), i(s, "dragenter", this._onDragEnter), i(s, "dragleave", this._onDragLeave), i(s, "drop", this._onDrop), this.enablePointerEvents || i(s, "touchstart", this._onTouchStart, c), F !== void 0 && r in F && (F[r](s, "gesture", this._onGesture), F[r](s, "drag", this._onDrag), F[r](s, "orientation", this._onOrientationChange), F[r](s, "shake", this._onShake), F[r](s, "longpress", this._onLongPress));
      }, removeListeners: function () {
        this.addOrRemove(a, "remove");
        var i = this._getEventPrefix();
        a(v.document, i + "up", this._onMouseUp), a(v.document, "touchend", this._onTouchEnd, c), a(v.document, i + "move", this._onMouseMove, c), a(v.document, "touchmove", this._onMouseMove, c);
      }, _bindEvents: function () {
        this.eventsBound || (this._onMouseDown = this._onMouseDown.bind(this), this._onTouchStart = this._onTouchStart.bind(this), this._onMouseMove = this._onMouseMove.bind(this), this._onMouseUp = this._onMouseUp.bind(this), this._onTouchEnd = this._onTouchEnd.bind(this), this._onResize = this._onResize.bind(this), this._onGesture = this._onGesture.bind(this), this._onDrag = this._onDrag.bind(this), this._onShake = this._onShake.bind(this), this._onLongPress = this._onLongPress.bind(this), this._onOrientationChange = this._onOrientationChange.bind(this), this._onMouseWheel = this._onMouseWheel.bind(this), this._onMouseOut = this._onMouseOut.bind(this), this._onMouseEnter = this._onMouseEnter.bind(this), this._onContextMenu = this._onContextMenu.bind(this), this._onDoubleClick = this._onDoubleClick.bind(this), this._onDragOver = this._onDragOver.bind(this), this._onDragEnter = this._simpleEventHandler.bind(this, "dragenter"), this._onDragLeave = this._simpleEventHandler.bind(this, "dragleave"), this._onDrop = this._simpleEventHandler.bind(this, "drop"), this.eventsBound = !0);
      }, _onGesture: function (i, r) {
        this.__onTransformGesture && this.__onTransformGesture(i, r);
      }, _onDrag: function (i, r) {
        this.__onDrag && this.__onDrag(i, r);
      }, _onMouseWheel: function (i) {
        this.__onMouseWheel(i);
      }, _onMouseOut: function (i) {
        var r = this._hoveredTarget;
        this.fire("mouse:out", { target: r, e: i }), this._hoveredTarget = null, r && r.fire("mouseout", { e: i });
        var s = this;
        this._hoveredTargets.forEach(function (u) {
          s.fire("mouse:out", { target: r, e: i }), u && r.fire("mouseout", { e: i });
        }), this._hoveredTargets = [], this._iTextInstances && this._iTextInstances.forEach(function (u) {
          u.isEditing && u.hiddenTextarea.focus();
        });
      }, _onMouseEnter: function (i) {
        this._currentTransform || this.findTarget(i) || (this.fire("mouse:over", { target: null, e: i }), this._hoveredTarget = null, this._hoveredTargets = []);
      }, _onOrientationChange: function (i, r) {
        this.__onOrientationChange && this.__onOrientationChange(i, r);
      }, _onShake: function (i, r) {
        this.__onShake && this.__onShake(i, r);
      }, _onLongPress: function (i, r) {
        this.__onLongPress && this.__onLongPress(i, r);
      }, _onDragOver: function (i) {
        i.preventDefault();
        var r = this._simpleEventHandler("dragover", i);
        this._fireEnterLeaveEvents(r, i);
      }, _onContextMenu: function (i) {
        return this.stopContextMenu && (i.stopPropagation(), i.preventDefault()), !1;
      }, _onDoubleClick: function (i) {
        this._cacheTransformEventData(i), this._handleEvent(i, "dblclick"), this._resetTransformEventData(i);
      }, getPointerId: function (i) {
        var r = i.changedTouches;
        return r ? r[0] && r[0].identifier : this.enablePointerEvents ? i.pointerId : -1;
      }, _isMainEvent: function (i) {
        return i.isPrimary === !0 || i.isPrimary !== !1 && (i.type === "touchend" && i.touches.length === 0 || !i.changedTouches || i.changedTouches[0].identifier === this.mainTouchId);
      }, _onTouchStart: function (i) {
        i.preventDefault(), this.mainTouchId === null && (this.mainTouchId = this.getPointerId(i)), this.__onMouseDown(i), this._resetTransformEventData();
        var r = this.upperCanvasEl, s = this._getEventPrefix();
        l(v.document, "touchend", this._onTouchEnd, c), l(v.document, "touchmove", this._onMouseMove, c), a(r, s + "down", this._onMouseDown);
      }, _onMouseDown: function (i) {
        this.__onMouseDown(i), this._resetTransformEventData();
        var r = this.upperCanvasEl, s = this._getEventPrefix();
        a(r, s + "move", this._onMouseMove, c), l(v.document, s + "up", this._onMouseUp), l(v.document, s + "move", this._onMouseMove, c);
      }, _onTouchEnd: function (i) {
        if (!(i.touches.length > 0)) {
          this.__onMouseUp(i), this._resetTransformEventData(), this.mainTouchId = null;
          var r = this._getEventPrefix();
          a(v.document, "touchend", this._onTouchEnd, c), a(v.document, "touchmove", this._onMouseMove, c);
          var s = this;
          this._willAddMouseDown && clearTimeout(this._willAddMouseDown), this._willAddMouseDown = setTimeout(function () {
            l(s.upperCanvasEl, r + "down", s._onMouseDown), s._willAddMouseDown = 0;
          }, 400);
        }
      }, _onMouseUp: function (i) {
        this.__onMouseUp(i), this._resetTransformEventData();
        var r = this.upperCanvasEl, s = this._getEventPrefix();
        this._isMainEvent(i) && (a(v.document, s + "up", this._onMouseUp), a(v.document, s + "move", this._onMouseMove, c), l(r, s + "move", this._onMouseMove, c));
      }, _onMouseMove: function (i) {
        !this.allowTouchScrolling && i.preventDefault && i.preventDefault(), this.__onMouseMove(i);
      }, _onResize: function () {
        this.calcOffset();
      }, _shouldRender: function (i) {
        var r = this._activeObject;
        return !!(!!r != !!i || r && i && r !== i) || (r && r.isEditing, !1);
      }, __onMouseUp: function (i) {
        var r, s = this._currentTransform, u = this._groupSelector, h = !1, p = !u || u.left === 0 && u.top === 0;
        if (this._cacheTransformEventData(i), r = this._target, this._handleEvent(i, "up:before"), f(i, 3)) this.fireRightClick && this._handleEvent(i, "up", 3, p);
        else {
          if (f(i, 2)) return this.fireMiddleClick && this._handleEvent(i, "up", 2, p), void this._resetTransformEventData();
          if (this.isDrawingMode && this._isCurrentlyDrawing) this._onMouseUpInDrawingMode(i);
          else if (this._isMainEvent(i)) {
            if (s && (this._finalizeCurrentTransform(i), h = s.actionPerformed), !p) {
              var m = r === this._activeObject;
              this._maybeGroupObjects(i), h || (h = this._shouldRender(r) || !m && r === this._activeObject);
            }
            if (r) {
              if (r.selectable && r !== this._activeObject && r.activeOn === "up") this.setActiveObject(r, i), h = !0;
              else {
                var y = r._findTargetCorner(this.getPointer(i, !0), v.util.isTouchEvent(i)), x = r.controls[y], S = x && x.getMouseUpHandler(i, r, x);
                if (S) {
                  var k = this.getPointer(i);
                  S(i, s, k.x, k.y);
                }
              }
              r.isMoving = !1;
            }
            this._setCursorFromEvent(i, r), this._handleEvent(i, "up", 1, p), this._groupSelector = null, this._currentTransform = null, r && (r.__corner = 0), h ? this.requestRenderAll() : p || this.renderTop();
          }
        }
      }, _simpleEventHandler: function (i, r) {
        var s = this.findTarget(r), u = this.targets, h = { e: r, target: s, subTargets: u };
        if (this.fire(i, h), s && s.fire(i, h), !u) return s;
        for (var p = 0; p < u.length; p++) u[p].fire(i, h);
        return s;
      }, _handleEvent: function (i, r, s, u) {
        var h = this._target, p = this.targets || [], m = { e: i, target: h, subTargets: p, button: s || 1, isClick: u || !1, pointer: this._pointer, absolutePointer: this._absolutePointer, transform: this._currentTransform };
        r === "up" && (m.currentTarget = this.findTarget(i), m.currentSubTargets = this.targets), this.fire("mouse:" + r, m), h && h.fire("mouse" + r, m);
        for (var y = 0; y < p.length; y++) p[y].fire("mouse" + r, m);
      }, _finalizeCurrentTransform: function (i) {
        var r, s = this._currentTransform, u = s.target, h = { e: i, target: u, transform: s, action: s.action };
        u._scaling && (u._scaling = !1), u.setCoords(), (s.actionPerformed || this.stateful && u.hasStateChanged()) && (s.actionPerformed && (r = this._addEventOptions(h, s), this._fire(r, h)), this._fire("modified", h));
      }, _addEventOptions: function (i, r) {
        var s, u;
        switch (r.action) {
          case "scaleX":
            s = "scaled", u = "x";
            break;
          case "scaleY":
            s = "scaled", u = "y";
            break;
          case "skewX":
            s = "skewed", u = "x";
            break;
          case "skewY":
            s = "skewed", u = "y";
            break;
          case "scale":
            s = "scaled", u = "equally";
            break;
          case "rotate":
            s = "rotated";
            break;
          case "drag":
            s = "moved";
        }
        return i.by = u, s;
      }, _onMouseDownInDrawingMode: function (i) {
        this._isCurrentlyDrawing = !0, this.getActiveObject() && this.discardActiveObject(i).requestRenderAll();
        var r = this.getPointer(i);
        this.freeDrawingBrush.onMouseDown(r, { e: i, pointer: r }), this._handleEvent(i, "down");
      }, _onMouseMoveInDrawingMode: function (i) {
        if (this._isCurrentlyDrawing) {
          var r = this.getPointer(i);
          this.freeDrawingBrush.onMouseMove(r, { e: i, pointer: r });
        }
        this.setCursor(this.freeDrawingCursor), this._handleEvent(i, "move");
      }, _onMouseUpInDrawingMode: function (i) {
        var r = this.getPointer(i);
        this._isCurrentlyDrawing = this.freeDrawingBrush.onMouseUp({ e: i, pointer: r }), this._handleEvent(i, "up");
      }, __onMouseDown: function (i) {
        this._cacheTransformEventData(i), this._handleEvent(i, "down:before");
        var r = this._target;
        if (f(i, 3)) this.fireRightClick && this._handleEvent(i, "down", 3);
        else if (f(i, 2)) this.fireMiddleClick && this._handleEvent(i, "down", 2);
        else if (this.isDrawingMode) this._onMouseDownInDrawingMode(i);
        else if (this._isMainEvent(i) && !this._currentTransform) {
          var s = this._pointer;
          this._previousPointer = s;
          var u = this._shouldRender(r), h = this._shouldGroup(i, r);
          if (this._shouldClearSelection(i, r) ? this.discardActiveObject(i) : h && (this._handleGrouping(i, r), r = this._activeObject), !this.selection || r && (r.selectable || r.isEditing || r === this._activeObject) || (this._groupSelector = { ex: this._absolutePointer.x, ey: this._absolutePointer.y, top: 0, left: 0 }), r) {
            var p = r === this._activeObject;
            r.selectable && r.activeOn === "down" && this.setActiveObject(r, i);
            var m = r._findTargetCorner(this.getPointer(i, !0), v.util.isTouchEvent(i));
            if (r.__corner = m, r === this._activeObject && (m || !h)) {
              this._setupCurrentTransform(i, r, p);
              var y = r.controls[m], x = (s = this.getPointer(i), y && y.getMouseDownHandler(i, r, y));
              x && x(i, this._currentTransform, s.x, s.y);
            }
          }
          this._handleEvent(i, "down"), (u || h) && this.requestRenderAll();
        }
      }, _resetTransformEventData: function () {
        this._target = null, this._pointer = null, this._absolutePointer = null;
      }, _cacheTransformEventData: function (i) {
        this._resetTransformEventData(), this._pointer = this.getPointer(i, !0), this._absolutePointer = this.restorePointerVpt(this._pointer), this._target = this._currentTransform ? this._currentTransform.target : this.findTarget(i) || null;
      }, _beforeTransform: function (i) {
        var r = this._currentTransform;
        this.stateful && r.target.saveState(), this.fire("before:transform", { e: i, transform: r });
      }, __onMouseMove: function (i) {
        var r, s;
        if (this._handleEvent(i, "move:before"), this._cacheTransformEventData(i), this.isDrawingMode) this._onMouseMoveInDrawingMode(i);
        else if (this._isMainEvent(i)) {
          var u = this._groupSelector;
          u ? (s = this._absolutePointer, u.left = s.x - u.ex, u.top = s.y - u.ey, this.renderTop()) : this._currentTransform ? this._transformObject(i) : (r = this.findTarget(i) || null, this._setCursorFromEvent(i, r), this._fireOverOutEvents(r, i)), this._handleEvent(i, "move"), this._resetTransformEventData();
        }
      }, _fireOverOutEvents: function (i, r) {
        var s = this._hoveredTarget, u = this._hoveredTargets, h = this.targets, p = Math.max(u.length, h.length);
        this.fireSyntheticInOutEvents(i, r, { oldTarget: s, evtOut: "mouseout", canvasEvtOut: "mouse:out", evtIn: "mouseover", canvasEvtIn: "mouse:over" });
        for (var m = 0; m < p; m++) this.fireSyntheticInOutEvents(h[m], r, { oldTarget: u[m], evtOut: "mouseout", evtIn: "mouseover" });
        this._hoveredTarget = i, this._hoveredTargets = this.targets.concat();
      }, _fireEnterLeaveEvents: function (i, r) {
        var s = this._draggedoverTarget, u = this._hoveredTargets, h = this.targets, p = Math.max(u.length, h.length);
        this.fireSyntheticInOutEvents(i, r, { oldTarget: s, evtOut: "dragleave", evtIn: "dragenter" });
        for (var m = 0; m < p; m++) this.fireSyntheticInOutEvents(h[m], r, { oldTarget: u[m], evtOut: "dragleave", evtIn: "dragenter" });
        this._draggedoverTarget = i;
      }, fireSyntheticInOutEvents: function (i, r, s) {
        var u, h, p, m = s.oldTarget, y = m !== i, x = s.canvasEvtIn, S = s.canvasEvtOut;
        y && (u = { e: r, target: i, previousTarget: m }, h = { e: r, target: m, nextTarget: i }), p = i && y, m && y && (S && this.fire(S, h), m.fire(s.evtOut, h)), p && (x && this.fire(x, u), i.fire(s.evtIn, u));
      }, __onMouseWheel: function (i) {
        this._cacheTransformEventData(i), this._handleEvent(i, "wheel"), this._resetTransformEventData();
      }, _transformObject: function (i) {
        var r = this.getPointer(i), s = this._currentTransform;
        s.reset = !1, s.shiftKey = i.shiftKey, s.altKey = i[this.centeredKey], this._performTransformAction(i, s, r), s.actionPerformed && this.requestRenderAll();
      }, _performTransformAction: function (i, r, s) {
        var u = s.x, h = s.y, p = r.action, m = !1, y = r.actionHandler;
        y && (m = y(i, r, u, h)), p === "drag" && m && (r.target.isMoving = !0, this.setCursor(r.target.moveCursor || this.moveCursor)), r.actionPerformed = r.actionPerformed || m;
      }, _fire: v.controlsUtils.fireEvent, _setCursorFromEvent: function (i, r) {
        if (!r) return this.setCursor(this.defaultCursor), !1;
        var s = r.hoverCursor || this.hoverCursor, u = this._activeObject && this._activeObject.type === "activeSelection" ? this._activeObject : null, h = (!u || !u.contains(r)) && r._findTargetCorner(this.getPointer(i, !0));
        h ? this.setCursor(this.getCornerCursor(h, r, i)) : (r.subTargetCheck && this.targets.concat().reverse().map(function (p) {
          s = p.hoverCursor || s;
        }), this.setCursor(s));
      }, getCornerCursor: function (i, r, s) {
        var u = r.controls[i];
        return u.cursorStyleHandler(s, u, r);
      }
    });
  }(), A = Math.min, tt = Math.max, v.util.object.extend(v.Canvas.prototype, {
    _shouldGroup: function (l, a) {
      var c = this._activeObject;
      return c && this._isSelectionKeyPressed(l) && a && a.selectable && this.selection && (c !== a || c.type === "activeSelection") && !a.onSelect({ e: l });
    }, _handleGrouping: function (l, a) {
      var c = this._activeObject;
      c.__corner || (a !== c || (a = this.findTarget(l, !0)) && a.selectable) && (c && c.type === "activeSelection" ? this._updateActiveSelection(a, l) : this._createActiveSelection(a, l));
    }, _updateActiveSelection: function (l, a) {
      var c = this._activeObject, f = c._objects.slice(0);
      c.contains(l) ? (c.removeWithUpdate(l), this._hoveredTarget = l, this._hoveredTargets = this.targets.concat(), c.size() === 1 && this._setActiveObject(c.item(0), a)) : (c.addWithUpdate(l), this._hoveredTarget = c, this._hoveredTargets = this.targets.concat()), this._fireSelectionEvents(f, a);
    }, _createActiveSelection: function (l, a) {
      var c = this.getActiveObjects(), f = this._createGroup(l);
      this._hoveredTarget = f, this._setActiveObject(f, a), this._fireSelectionEvents(c, a);
    }, _createGroup: function (l) {
      var a = this._objects, c = a.indexOf(this._activeObject) < a.indexOf(l) ? [this._activeObject, l] : [l, this._activeObject];
      return this._activeObject.isEditing && this._activeObject.exitEditing(), new v.ActiveSelection(c, { canvas: this });
    }, _groupSelectedObjects: function (l) {
      var a, c = this._collectObjects(l);
      c.length === 1 ? this.setActiveObject(c[0], l) : c.length > 1 && (a = new v.ActiveSelection(c.reverse(), { canvas: this }), this.setActiveObject(a, l));
    }, _collectObjects: function (l) {
      for (var a, c = [], f = this._groupSelector.ex, i = this._groupSelector.ey, r = f + this._groupSelector.left, s = i + this._groupSelector.top, u = new v.Point(A(f, r), A(i, s)), h = new v.Point(tt(f, r), tt(i, s)), p = !this.selectionFullyContained, m = f === r && i === s, y = this._objects.length; y-- && !((a = this._objects[y]) && a.selectable && a.visible && (p && a.intersectsWithRect(u, h, !0) || a.isContainedWithinRect(u, h, !0) || p && a.containsPoint(u, null, !0) || p && a.containsPoint(h, null, !0)) && (c.push(a), m)););
      return c.length > 1 && (c = c.filter(function (x) {
        return !x.onSelect({ e: l });
      })), c;
    }, _maybeGroupObjects: function (l) {
      this.selection && this._groupSelector && this._groupSelectedObjects(l), this.setCursor(this.defaultCursor), this._groupSelector = null;
    }
  }), v.util.object.extend(v.StaticCanvas.prototype, {
    toDataURL: function (l) {
      l || (l = {});
      var a = l.format || "png", c = l.quality || 1, f = (l.multiplier || 1) * (l.enableRetinaScaling ? this.getRetinaScaling() : 1), i = this.toCanvasElement(f, l);
      return v.util.toDataURL(i, a, c);
    }, toCanvasElement: function (l, a) {
      l = l || 1;
      var c = ((a = a || {}).width || this.width) * l, f = (a.height || this.height) * l, i = this.getZoom(), r = this.width, s = this.height, u = i * l, h = this.viewportTransform, p = (h[4] - (a.left || 0)) * l, m = (h[5] - (a.top || 0)) * l, y = this.interactive, x = [u, 0, 0, u, p, m], S = this.enableRetinaScaling, k = v.util.createCanvasElement(), D = this.contextTop;
      return k.width = c, k.height = f, this.contextTop = null, this.enableRetinaScaling = !1, this.interactive = !1, this.viewportTransform = x, this.width = c, this.height = f, this.calcViewportBoundaries(), this.renderCanvas(k.getContext("2d"), this._objects), this.viewportTransform = h, this.width = r, this.height = s, this.calcViewportBoundaries(), this.interactive = y, this.enableRetinaScaling = S, this.contextTop = D, k;
    }
  }), v.util.object.extend(v.StaticCanvas.prototype, {
    loadFromJSON: function (l, a, c) {
      if (l) {
        var f = typeof l == "string" ? JSON.parse(l) : v.util.object.clone(l), i = this, r = f.clipPath, s = this.renderOnAddRemove;
        return this.renderOnAddRemove = !1, delete f.clipPath, this._enlivenObjects(f.objects, function (u) {
          i.clear(), i._setBgOverlay(f, function () {
            r ? i._enlivenObjects([r], function (h) {
              i.clipPath = h[0], i.__setupCanvas.call(i, f, u, s, a);
            }) : i.__setupCanvas.call(i, f, u, s, a);
          });
        }, c), this;
      }
    }, __setupCanvas: function (l, a, c, f) {
      var i = this;
      a.forEach(function (r, s) {
        i.insertAt(r, s);
      }), this.renderOnAddRemove = c, delete l.objects, delete l.backgroundImage, delete l.overlayImage, delete l.background, delete l.overlay, this._setOptions(l), this.renderAll(), f && f();
    }, _setBgOverlay: function (l, a) {
      var c = { backgroundColor: !1, overlayColor: !1, backgroundImage: !1, overlayImage: !1 };
      if (l.backgroundImage || l.overlayImage || l.background || l.overlay) {
        var f = function () {
          c.backgroundImage && c.overlayImage && c.backgroundColor && c.overlayColor && a && a();
        };
        this.__setBgOverlay("backgroundImage", l.backgroundImage, c, f), this.__setBgOverlay("overlayImage", l.overlayImage, c, f), this.__setBgOverlay("backgroundColor", l.background, c, f), this.__setBgOverlay("overlayColor", l.overlay, c, f);
      } else a && a();
    }, __setBgOverlay: function (l, a, c, f) {
      var i = this;
      if (!a) return c[l] = !0, void (f && f());
      l === "backgroundImage" || l === "overlayImage" ? v.util.enlivenObjects([a], function (r) {
        i[l] = r[0], c[l] = !0, f && f();
      }) : this["set" + v.util.string.capitalize(l, !0)](a, function () {
        c[l] = !0, f && f();
      });
    }, _enlivenObjects: function (l, a, c) {
      l && l.length !== 0 ? v.util.enlivenObjects(l, function (f) {
        a && a(f);
      }, null, c) : a && a([]);
    }, _toDataURL: function (l, a) {
      this.clone(function (c) {
        a(c.toDataURL(l));
      });
    }, _toDataURLWithMultiplier: function (l, a, c) {
      this.clone(function (f) {
        c(f.toDataURLWithMultiplier(l, a));
      });
    }, clone: function (l, a) {
      var c = JSON.stringify(this.toJSON(a));
      this.cloneWithoutData(function (f) {
        f.loadFromJSON(c, function () {
          l && l(f);
        });
      });
    }, cloneWithoutData: function (l) {
      var a = v.util.createCanvasElement();
      a.width = this.width, a.height = this.height;
      var c = new v.Canvas(a);
      this.backgroundImage ? (c.setBackgroundImage(this.backgroundImage.src, function () {
        c.renderAll(), l && l(c);
      }), c.backgroundImageOpacity = this.backgroundImageOpacity, c.backgroundImageStretch = this.backgroundImageStretch) : l && l(c);
    }
  }), dt = v.util.degreesToRadians, it = v.util.radiansToDegrees, v.util.object.extend(v.Canvas.prototype, {
    __onTransformGesture: function (l, a) {
      if (!this.isDrawingMode && l.touches && l.touches.length === 2 && a.gesture === "gesture") {
        var c = this.findTarget(l);
        c !== void 0 && (this.__gesturesParams = { e: l, self: a, target: c }, this.__gesturesRenderer()), this.fire("touch:gesture", { target: c, e: l, self: a });
      }
    }, __gesturesParams: null, __gesturesRenderer: function () {
      if (this.__gesturesParams !== null && this._currentTransform !== null) {
        var l = this.__gesturesParams.self, a = this._currentTransform, c = this.__gesturesParams.e;
        a.action = "scale", a.originX = a.originY = "center", this._scaleObjectBy(l.scale, c), l.rotation !== 0 && (a.action = "rotate", this._rotateObjectByAngle(l.rotation, c)), this.requestRenderAll(), a.action = "drag";
      }
    }, __onDrag: function (l, a) {
      this.fire("touch:drag", { e: l, self: a });
    }, __onOrientationChange: function (l, a) {
      this.fire("touch:orientation", { e: l, self: a });
    }, __onShake: function (l, a) {
      this.fire("touch:shake", { e: l, self: a });
    }, __onLongPress: function (l, a) {
      this.fire("touch:longpress", { e: l, self: a });
    }, _scaleObjectBy: function (l, a) {
      var c = this._currentTransform, f = c.target;
      return c.gestureScale = l, f._scaling = !0, v.controlsUtils.scalingEqually(a, c, 0, 0);
    }, _rotateObjectByAngle: function (l, a) {
      var c = this._currentTransform;
      c.target.get("lockRotation") || (c.target.rotate(it(dt(l) + c.theta)), this._fire("rotating", { target: c.target, e: a, transform: c }));
    }
  }), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.object.extend, f = a.util.object.clone, i = a.util.toFixed, r = a.util.string.capitalize, s = a.util.degreesToRadians, u = !a.isLikelyNode;
    a.Object || (a.Object = a.util.createClass(a.CommonMethods, {
      type: "object", originX: "left", originY: "top", top: 0, left: 0, width: 0, height: 0, scaleX: 1, scaleY: 1, flipX: !1, flipY: !1, opacity: 1, angle: 0, skewX: 0, skewY: 0, cornerSize: 13, touchCornerSize: 24, transparentCorners: !0, hoverCursor: null, moveCursor: null, padding: 0, borderColor: "rgb(178,204,255)", borderDashArray: null, cornerColor: "rgb(178,204,255)", cornerStrokeColor: null, cornerStyle: "rect", cornerDashArray: null, centeredScaling: !1, centeredRotation: !0, fill: "rgb(0,0,0)", fillRule: "nonzero", globalCompositeOperation: "source-over", backgroundColor: "", selectionBackgroundColor: "", stroke: null, strokeWidth: 1, strokeDashArray: null, strokeDashOffset: 0, strokeLineCap: "butt", strokeLineJoin: "miter", strokeMiterLimit: 4, shadow: null, borderOpacityWhenMoving: 0.4, borderScaleFactor: 1, minScaleLimit: 0, selectable: !0, evented: !0, visible: !0, hasControls: !0, hasBorders: !0, perPixelTargetFind: !1, includeDefaultValues: !0, lockMovementX: !1, lockMovementY: !1, lockRotation: !1, lockScalingX: !1, lockScalingY: !1, lockSkewingX: !1, lockSkewingY: !1, lockScalingFlip: !1, excludeFromExport: !1, objectCaching: u, statefullCache: !1, noScaleCache: !0, strokeUniform: !1, dirty: !0, __corner: 0, paintFirst: "fill", activeOn: "down", stateProperties: "top left width height scaleX scaleY flipX flipY originX originY transformMatrix stroke strokeWidth strokeDashArray strokeLineCap strokeDashOffset strokeLineJoin strokeMiterLimit angle opacity fill globalCompositeOperation shadow visible backgroundColor skewX skewY fillRule paintFirst clipPath strokeUniform".split(" "), cacheProperties: "fill stroke strokeWidth strokeDashArray width height paintFirst strokeUniform strokeLineCap strokeDashOffset strokeLineJoin strokeMiterLimit backgroundColor clipPath".split(" "), colorProperties: "fill stroke backgroundColor".split(" "), clipPath: void 0, inverted: !1, absolutePositioned: !1, initialize: function (h) {
        h && this.setOptions(h);
      }, _createCacheCanvas: function () {
        this._cacheProperties = {}, this._cacheCanvas = a.util.createCanvasElement(), this._cacheContext = this._cacheCanvas.getContext("2d"), this._updateCacheCanvas(), this.dirty = !0;
      }, _limitCacheSize: function (h) {
        var p = a.perfLimitSizeTotal, m = h.width, y = h.height, x = a.maxCacheSideLimit, S = a.minCacheSideLimit;
        if (m <= x && y <= x && m * y <= p) return m < S && (h.width = S), y < S && (h.height = S), h;
        var k = m / y, D = a.util.limitDimsByArea(k, p), M = a.util.capValue, C = M(S, D.x, x), O = M(S, D.y, x);
        return m > C && (h.zoomX /= m / C, h.width = C, h.capped = !0), y > O && (h.zoomY /= y / O, h.height = O, h.capped = !0), h;
      }, _getCacheCanvasDimensions: function () {
        var h = this.getTotalObjectScaling(), p = this._getTransformedDimensions(0, 0), m = p.x * h.scaleX / this.scaleX, y = p.y * h.scaleY / this.scaleY;
        return { width: m + 2, height: y + 2, zoomX: h.scaleX, zoomY: h.scaleY, x: m, y };
      }, _updateCacheCanvas: function () {
        var h = this.canvas;
        if (this.noScaleCache && h && h._currentTransform) {
          var p = h._currentTransform.target, m = h._currentTransform.action;
          if (this === p && m.slice && m.slice(0, 5) === "scale") return !1;
        }
        var y, x, S = this._cacheCanvas, k = this._limitCacheSize(this._getCacheCanvasDimensions()), D = a.minCacheSideLimit, M = k.width, C = k.height, O = k.zoomX, N = k.zoomY, W = M !== this.cacheWidth || C !== this.cacheHeight, X = this.zoomX !== O || this.zoomY !== N, I = W || X, T = 0, R = 0, Q = !1;
        if (W) {
          var K = this._cacheCanvas.width, U = this._cacheCanvas.height, B = M > K || C > U;
          Q = B || (M < 0.9 * K || C < 0.9 * U) && K > D && U > D, B && !k.capped && (M > D || C > D) && (T = 0.1 * M, R = 0.1 * C);
        }
        return this instanceof a.Text && this.path && (I = !0, Q = !0, T += this.getHeightOfLine(0) * this.zoomX, R += this.getHeightOfLine(0) * this.zoomY), !!I && (Q ? (S.width = Math.ceil(M + T), S.height = Math.ceil(C + R)) : (this._cacheContext.setTransform(1, 0, 0, 1, 0, 0), this._cacheContext.clearRect(0, 0, S.width, S.height)), y = k.x / 2, x = k.y / 2, this.cacheTranslationX = Math.round(S.width / 2 - y) + y, this.cacheTranslationY = Math.round(S.height / 2 - x) + x, this.cacheWidth = M, this.cacheHeight = C, this._cacheContext.translate(this.cacheTranslationX, this.cacheTranslationY), this._cacheContext.scale(O, N), this.zoomX = O, this.zoomY = N, !0);
      }, setOptions: function (h) {
        this._setOptions(h), this._initGradient(h.fill, "fill"), this._initGradient(h.stroke, "stroke"), this._initPattern(h.fill, "fill"), this._initPattern(h.stroke, "stroke");
      }, transform: function (h) {
        var p = this.group && !this.group._transformDone || this.group && this.canvas && h === this.canvas.contextTop, m = this.calcTransformMatrix(!p);
        h.transform(m[0], m[1], m[2], m[3], m[4], m[5]);
      }, toObject: function (h) {
        var p = a.Object.NUM_FRACTION_DIGITS, m = { type: this.type, version: a.version, originX: this.originX, originY: this.originY, left: i(this.left, p), top: i(this.top, p), width: i(this.width, p), height: i(this.height, p), fill: this.fill && this.fill.toObject ? this.fill.toObject() : this.fill, stroke: this.stroke && this.stroke.toObject ? this.stroke.toObject() : this.stroke, strokeWidth: i(this.strokeWidth, p), strokeDashArray: this.strokeDashArray ? this.strokeDashArray.concat() : this.strokeDashArray, strokeLineCap: this.strokeLineCap, strokeDashOffset: this.strokeDashOffset, strokeLineJoin: this.strokeLineJoin, strokeUniform: this.strokeUniform, strokeMiterLimit: i(this.strokeMiterLimit, p), scaleX: i(this.scaleX, p), scaleY: i(this.scaleY, p), angle: i(this.angle, p), flipX: this.flipX, flipY: this.flipY, opacity: i(this.opacity, p), shadow: this.shadow && this.shadow.toObject ? this.shadow.toObject() : this.shadow, visible: this.visible, backgroundColor: this.backgroundColor, fillRule: this.fillRule, paintFirst: this.paintFirst, globalCompositeOperation: this.globalCompositeOperation, skewX: i(this.skewX, p), skewY: i(this.skewY, p) };
        return this.clipPath && !this.clipPath.excludeFromExport && (m.clipPath = this.clipPath.toObject(h), m.clipPath.inverted = this.clipPath.inverted, m.clipPath.absolutePositioned = this.clipPath.absolutePositioned), a.util.populateWithProperties(this, m, h), this.includeDefaultValues || (m = this._removeDefaultValues(m)), m;
      }, toDatalessObject: function (h) {
        return this.toObject(h);
      }, _removeDefaultValues: function (h) {
        var p = a.util.getKlass(h.type).prototype;
        return p.stateProperties.forEach(function (m) {
          m !== "left" && m !== "top" && (h[m] === p[m] && delete h[m], Object.prototype.toString.call(h[m]) === "[object Array]" && Object.prototype.toString.call(p[m]) === "[object Array]" && h[m].length === 0 && p[m].length === 0 && delete h[m]);
        }), h;
      }, toString: function () {
        return "#<fabric." + r(this.type) + ">";
      }, getObjectScaling: function () {
        if (!this.group) return { scaleX: this.scaleX, scaleY: this.scaleY };
        var h = a.util.qrDecompose(this.calcTransformMatrix());
        return { scaleX: Math.abs(h.scaleX), scaleY: Math.abs(h.scaleY) };
      }, getTotalObjectScaling: function () {
        var h = this.getObjectScaling(), p = h.scaleX, m = h.scaleY;
        if (this.canvas) {
          var y = this.canvas.getZoom(), x = this.canvas.getRetinaScaling();
          p *= y * x, m *= y * x;
        }
        return { scaleX: p, scaleY: m };
      }, getObjectOpacity: function () {
        var h = this.opacity;
        return this.group && (h *= this.group.getObjectOpacity()), h;
      }, _set: function (h, p) {
        var m = h === "scaleX" || h === "scaleY", y = this[h] !== p, x = !1;
        return m && (p = this._constrainScale(p)), h === "scaleX" && p < 0 ? (this.flipX = !this.flipX, p *= -1) : h === "scaleY" && p < 0 ? (this.flipY = !this.flipY, p *= -1) : h !== "shadow" || !p || p instanceof a.Shadow ? h === "dirty" && this.group && this.group.set("dirty", p) : p = new a.Shadow(p), this[h] = p, y && (x = this.group && this.group.isOnACache(), this.cacheProperties.indexOf(h) > -1 ? (this.dirty = !0, x && this.group.set("dirty", !0)) : x && this.stateProperties.indexOf(h) > -1 && this.group.set("dirty", !0)), this;
      }, setOnGroup: function () {
      }, getViewportTransform: function () {
        return this.canvas && this.canvas.viewportTransform ? this.canvas.viewportTransform : a.iMatrix.concat();
      }, isNotVisible: function () {
        return this.opacity === 0 || !this.width && !this.height && this.strokeWidth === 0 || !this.visible;
      }, render: function (h) {
        this.isNotVisible() || this.canvas && this.canvas.skipOffscreen && !this.group && !this.isOnScreen() || (h.save(), this._setupCompositeOperation(h), this.drawSelectionBackground(h), this.transform(h), this._setOpacity(h), this._setShadow(h, this), this.shouldCache() ? (this.renderCache(), this.drawCacheOnCanvas(h)) : (this._removeCacheCanvas(), this.dirty = !1, this.drawObject(h), this.objectCaching && this.statefullCache && this.saveState({ propertySet: "cacheProperties" })), h.restore());
      }, renderCache: function (h) {
        h = h || {}, this._cacheCanvas || this._createCacheCanvas(), this.isCacheDirty() && (this.statefullCache && this.saveState({ propertySet: "cacheProperties" }), this.drawObject(this._cacheContext, h.forClipping), this.dirty = !1);
      }, _removeCacheCanvas: function () {
        this._cacheCanvas = null, this.cacheWidth = 0, this.cacheHeight = 0;
      }, hasStroke: function () {
        return this.stroke && this.stroke !== "transparent" && this.strokeWidth !== 0;
      }, hasFill: function () {
        return this.fill && this.fill !== "transparent";
      }, needsItsOwnCache: function () {
        return !(this.paintFirst !== "stroke" || !this.hasFill() || !this.hasStroke() || typeof this.shadow != "object") || !!this.clipPath;
      }, shouldCache: function () {
        return this.ownCaching = this.needsItsOwnCache() || this.objectCaching && (!this.group || !this.group.isOnACache()), this.ownCaching;
      }, willDrawShadow: function () {
        return !!this.shadow && (this.shadow.offsetX !== 0 || this.shadow.offsetY !== 0);
      }, drawClipPathOnCache: function (h) {
        var p = this.clipPath;
        if (h.save(), p.inverted ? h.globalCompositeOperation = "destination-out" : h.globalCompositeOperation = "destination-in", p.absolutePositioned) {
          var m = a.util.invertTransform(this.calcTransformMatrix());
          h.transform(m[0], m[1], m[2], m[3], m[4], m[5]);
        }
        p.transform(h), h.scale(1 / p.zoomX, 1 / p.zoomY), h.drawImage(p._cacheCanvas, -p.cacheTranslationX, -p.cacheTranslationY), h.restore();
      }, drawObject: function (h, p) {
        var m = this.fill, y = this.stroke;
        p ? (this.fill = "black", this.stroke = "", this._setClippingProperties(h)) : this._renderBackground(h), this._render(h), this._drawClipPath(h), this.fill = m, this.stroke = y;
      }, _drawClipPath: function (h) {
        var p = this.clipPath;
        p && (p.canvas = this.canvas, p.shouldCache(), p._transformDone = !0, p.renderCache({ forClipping: !0 }), this.drawClipPathOnCache(h));
      }, drawCacheOnCanvas: function (h) {
        h.scale(1 / this.zoomX, 1 / this.zoomY), h.drawImage(this._cacheCanvas, -this.cacheTranslationX, -this.cacheTranslationY);
      }, isCacheDirty: function (h) {
        if (this.isNotVisible()) return !1;
        if (this._cacheCanvas && !h && this._updateCacheCanvas()) return !0;
        if (this.dirty || this.clipPath && this.clipPath.absolutePositioned || this.statefullCache && this.hasStateChanged("cacheProperties")) {
          if (this._cacheCanvas && !h) {
            var p = this.cacheWidth / this.zoomX, m = this.cacheHeight / this.zoomY;
            this._cacheContext.clearRect(-p / 2, -m / 2, p, m);
          }
          return !0;
        }
        return !1;
      }, _renderBackground: function (h) {
        if (this.backgroundColor) {
          var p = this._getNonTransformedDimensions();
          h.fillStyle = this.backgroundColor, h.fillRect(-p.x / 2, -p.y / 2, p.x, p.y), this._removeShadow(h);
        }
      }, _setOpacity: function (h) {
        this.group && !this.group._transformDone ? h.globalAlpha = this.getObjectOpacity() : h.globalAlpha *= this.opacity;
      }, _setStrokeStyles: function (h, p) {
        var m = p.stroke;
        m && (h.lineWidth = p.strokeWidth, h.lineCap = p.strokeLineCap, h.lineDashOffset = p.strokeDashOffset, h.lineJoin = p.strokeLineJoin, h.miterLimit = p.strokeMiterLimit, m.toLive ? m.gradientUnits === "percentage" || m.gradientTransform || m.patternTransform ? this._applyPatternForTransformedGradient(h, m) : (h.strokeStyle = m.toLive(h, this), this._applyPatternGradientTransform(h, m)) : h.strokeStyle = p.stroke);
      }, _setFillStyles: function (h, p) {
        var m = p.fill;
        m && (m.toLive ? (h.fillStyle = m.toLive(h, this), this._applyPatternGradientTransform(h, p.fill)) : h.fillStyle = m);
      }, _setClippingProperties: function (h) {
        h.globalAlpha = 1, h.strokeStyle = "transparent", h.fillStyle = "#000000";
      }, _setLineDash: function (h, p) {
        p && p.length !== 0 && (1 & p.length && p.push.apply(p, p), h.setLineDash(p));
      }, _renderControls: function (h, p) {
        var m, y, x, S = this.getViewportTransform(), k = this.calcTransformMatrix();
        y = (p = p || {}).hasBorders !== void 0 ? p.hasBorders : this.hasBorders, x = p.hasControls !== void 0 ? p.hasControls : this.hasControls, k = a.util.multiplyTransformMatrices(S, k), m = a.util.qrDecompose(k), h.save(), h.translate(m.translateX, m.translateY), h.lineWidth = 1 * this.borderScaleFactor, this.group || (h.globalAlpha = this.isMoving ? this.borderOpacityWhenMoving : 1), h.rotate(s(m.angle)), p.forActiveSelection || this.group ? y && this.drawBordersInGroup(h, m, p) : y && this.drawBorders(h, p), x && this.drawControls(h, p), h.restore();
      }, _setShadow: function (h) {
        if (this.shadow) {
          var p, m = this.shadow, y = this.canvas, x = y && y.viewportTransform[0] || 1, S = y && y.viewportTransform[3] || 1;
          p = m.nonScaling ? { scaleX: 1, scaleY: 1 } : this.getObjectScaling(), y && y._isRetinaScaling() && (x *= a.devicePixelRatio, S *= a.devicePixelRatio), h.shadowColor = m.color, h.shadowBlur = m.blur * a.browserShadowBlurConstant * (x + S) * (p.scaleX + p.scaleY) / 4, h.shadowOffsetX = m.offsetX * x * p.scaleX, h.shadowOffsetY = m.offsetY * S * p.scaleY;
        }
      }, _removeShadow: function (h) {
        this.shadow && (h.shadowColor = "", h.shadowBlur = h.shadowOffsetX = h.shadowOffsetY = 0);
      }, _applyPatternGradientTransform: function (h, p) {
        if (!p || !p.toLive) return { offsetX: 0, offsetY: 0 };
        var m = p.gradientTransform || p.patternTransform, y = -this.width / 2 + p.offsetX || 0, x = -this.height / 2 + p.offsetY || 0;
        return p.gradientUnits === "percentage" ? h.transform(this.width, 0, 0, this.height, y, x) : h.transform(1, 0, 0, 1, y, x), m && h.transform(m[0], m[1], m[2], m[3], m[4], m[5]), { offsetX: y, offsetY: x };
      }, _renderPaintInOrder: function (h) {
        this.paintFirst === "stroke" ? (this._renderStroke(h), this._renderFill(h)) : (this._renderFill(h), this._renderStroke(h));
      }, _render: function () {
      }, _renderFill: function (h) {
        this.fill && (h.save(), this._setFillStyles(h, this), this.fillRule === "evenodd" ? h.fill("evenodd") : h.fill(), h.restore());
      }, _renderStroke: function (h) {
        if (this.stroke && this.strokeWidth !== 0) {
          if (this.shadow && !this.shadow.affectStroke && this._removeShadow(h), h.save(), this.strokeUniform && this.group) {
            var p = this.getObjectScaling();
            h.scale(1 / p.scaleX, 1 / p.scaleY);
          } else this.strokeUniform && h.scale(1 / this.scaleX, 1 / this.scaleY);
          this._setLineDash(h, this.strokeDashArray), this._setStrokeStyles(h, this), h.stroke(), h.restore();
        }
      }, _applyPatternForTransformedGradient: function (h, p) {
        var m, y = this._limitCacheSize(this._getCacheCanvasDimensions()), x = a.util.createCanvasElement(), S = this.canvas.getRetinaScaling(), k = y.x / this.scaleX / S, D = y.y / this.scaleY / S;
        x.width = k, x.height = D, (m = x.getContext("2d")).beginPath(), m.moveTo(0, 0), m.lineTo(k, 0), m.lineTo(k, D), m.lineTo(0, D), m.closePath(), m.translate(k / 2, D / 2), m.scale(y.zoomX / this.scaleX / S, y.zoomY / this.scaleY / S), this._applyPatternGradientTransform(m, p), m.fillStyle = p.toLive(h), m.fill(), h.translate(-this.width / 2 - this.strokeWidth / 2, -this.height / 2 - this.strokeWidth / 2), h.scale(S * this.scaleX / y.zoomX, S * this.scaleY / y.zoomY), h.strokeStyle = m.createPattern(x, "no-repeat");
      }, _findCenterFromElement: function () {
        return { x: this.left + this.width / 2, y: this.top + this.height / 2 };
      }, _assignTransformMatrixProps: function () {
        if (this.transformMatrix) {
          var h = a.util.qrDecompose(this.transformMatrix);
          this.flipX = !1, this.flipY = !1, this.set("scaleX", h.scaleX), this.set("scaleY", h.scaleY), this.angle = h.angle, this.skewX = h.skewX, this.skewY = 0;
        }
      }, _removeTransformMatrix: function (h) {
        var p = this._findCenterFromElement();
        this.transformMatrix && (this._assignTransformMatrixProps(), p = a.util.transformPoint(p, this.transformMatrix)), this.transformMatrix = null, h && (this.scaleX *= h.scaleX, this.scaleY *= h.scaleY, this.cropX = h.cropX, this.cropY = h.cropY, p.x += h.offsetLeft, p.y += h.offsetTop, this.width = h.width, this.height = h.height), this.setPositionByOrigin(p, "center", "center");
      }, clone: function (h, p) {
        var m = this.toObject(p);
        this.constructor.fromObject ? this.constructor.fromObject(m, h) : a.Object._fromObject("Object", m, h);
      }, cloneAsImage: function (h, p) {
        var m = this.toCanvasElement(p);
        return h && h(new a.Image(m)), this;
      }, toCanvasElement: function (h) {
        h || (h = {});
        var p = a.util, m = p.saveObjectTransform(this), y = this.group, x = this.shadow, S = Math.abs, k = (h.multiplier || 1) * (h.enableRetinaScaling ? a.devicePixelRatio : 1);
        delete this.group, h.withoutTransform && p.resetObjectTransform(this), h.withoutShadow && (this.shadow = null);
        var D, M, C, O, N = a.util.createCanvasElement(), W = this.getBoundingRect(!0, !0), X = this.shadow, I = { x: 0, y: 0 };
        X && (M = X.blur, D = X.nonScaling ? { scaleX: 1, scaleY: 1 } : this.getObjectScaling(), I.x = 2 * Math.round(S(X.offsetX) + M) * S(D.scaleX), I.y = 2 * Math.round(S(X.offsetY) + M) * S(D.scaleY)), C = W.width + I.x, O = W.height + I.y, N.width = Math.ceil(C), N.height = Math.ceil(O);
        var T = new a.StaticCanvas(N, { enableRetinaScaling: !1, renderOnAddRemove: !1, skipOffscreen: !1 });
        h.format === "jpeg" && (T.backgroundColor = "#fff"), this.setPositionByOrigin(new a.Point(T.width / 2, T.height / 2), "center", "center");
        var R = this.canvas;
        T.add(this);
        var Q = T.toCanvasElement(k || 1, h);
        return this.shadow = x, this.set("canvas", R), y && (this.group = y), this.set(m).setCoords(), T._objects = [], T.dispose(), T = null, Q;
      }, toDataURL: function (h) {
        return h || (h = {}), a.util.toDataURL(this.toCanvasElement(h), h.format || "png", h.quality || 1);
      }, isType: function (h) {
        return this.type === h;
      }, complexity: function () {
        return 1;
      }, toJSON: function (h) {
        return this.toObject(h);
      }, rotate: function (h) {
        var p = (this.originX !== "center" || this.originY !== "center") && this.centeredRotation;
        return p && this._setOriginToCenter(), this.set("angle", h), p && this._resetOrigin(), this;
      }, centerH: function () {
        return this.canvas && this.canvas.centerObjectH(this), this;
      }, viewportCenterH: function () {
        return this.canvas && this.canvas.viewportCenterObjectH(this), this;
      }, centerV: function () {
        return this.canvas && this.canvas.centerObjectV(this), this;
      }, viewportCenterV: function () {
        return this.canvas && this.canvas.viewportCenterObjectV(this), this;
      }, center: function () {
        return this.canvas && this.canvas.centerObject(this), this;
      }, viewportCenter: function () {
        return this.canvas && this.canvas.viewportCenterObject(this), this;
      }, getLocalPointer: function (h, p) {
        p = p || this.canvas.getPointer(h);
        var m = new a.Point(p.x, p.y), y = this._getLeftTopCoords();
        return this.angle && (m = a.util.rotatePoint(m, y, s(-this.angle))), { x: m.x - y.x, y: m.y - y.y };
      }, _setupCompositeOperation: function (h) {
        this.globalCompositeOperation && (h.globalCompositeOperation = this.globalCompositeOperation);
      }
    }), a.util.createAccessors && a.util.createAccessors(a.Object), c(a.Object.prototype, a.Observable), a.Object.NUM_FRACTION_DIGITS = 2, a.Object._fromObject = function (h, p, m, y) {
      var x = a[h];
      p = f(p, !0), a.util.enlivenPatterns([p.fill, p.stroke], function (S) {
        S[0] !== void 0 && (p.fill = S[0]), S[1] !== void 0 && (p.stroke = S[1]), a.util.enlivenObjects([p.clipPath], function (k) {
          p.clipPath = k[0];
          var D = y ? new x(p[y], p) : new x(p);
          m && m(D);
        });
      });
    }, a.Object.__uid = 0);
  }(t), function () {
    var l = v.util.degreesToRadians, a = { left: -0.5, center: 0, right: 0.5 }, c = { top: -0.5, center: 0, bottom: 0.5 };
    v.util.object.extend(v.Object.prototype, {
      translateToGivenOrigin: function (f, i, r, s, u) {
        var h, p, m, y = f.x, x = f.y;
        return typeof i == "string" ? i = a[i] : i -= 0.5, typeof s == "string" ? s = a[s] : s -= 0.5, typeof r == "string" ? r = c[r] : r -= 0.5, typeof u == "string" ? u = c[u] : u -= 0.5, p = u - r, ((h = s - i) || p) && (m = this._getTransformedDimensions(), y = f.x + h * m.x, x = f.y + p * m.y), new v.Point(y, x);
      }, translateToCenterPoint: function (f, i, r) {
        var s = this.translateToGivenOrigin(f, i, r, "center", "center");
        return this.angle ? v.util.rotatePoint(s, f, l(this.angle)) : s;
      }, translateToOriginPoint: function (f, i, r) {
        var s = this.translateToGivenOrigin(f, "center", "center", i, r);
        return this.angle ? v.util.rotatePoint(s, f, l(this.angle)) : s;
      }, getCenterPoint: function () {
        var f = new v.Point(this.left, this.top);
        return this.translateToCenterPoint(f, this.originX, this.originY);
      }, getPointByOrigin: function (f, i) {
        var r = this.getCenterPoint();
        return this.translateToOriginPoint(r, f, i);
      }, toLocalPoint: function (f, i, r) {
        var s, u, h = this.getCenterPoint();
        return s = i !== void 0 && r !== void 0 ? this.translateToGivenOrigin(h, "center", "center", i, r) : new v.Point(this.left, this.top), u = new v.Point(f.x, f.y), this.angle && (u = v.util.rotatePoint(u, h, -l(this.angle))), u.subtractEquals(s);
      }, setPositionByOrigin: function (f, i, r) {
        var s = this.translateToCenterPoint(f, i, r), u = this.translateToOriginPoint(s, this.originX, this.originY);
        this.set("left", u.x), this.set("top", u.y);
      }, adjustPosition: function (f) {
        var i, r, s = l(this.angle), u = this.getScaledWidth(), h = v.util.cos(s) * u, p = v.util.sin(s) * u;
        i = typeof this.originX == "string" ? a[this.originX] : this.originX - 0.5, r = typeof f == "string" ? a[f] : f - 0.5, this.left += h * (r - i), this.top += p * (r - i), this.setCoords(), this.originX = f;
      }, _setOriginToCenter: function () {
        this._originalOriginX = this.originX, this._originalOriginY = this.originY;
        var f = this.getCenterPoint();
        this.originX = "center", this.originY = "center", this.left = f.x, this.top = f.y;
      }, _resetOrigin: function () {
        var f = this.translateToOriginPoint(this.getCenterPoint(), this._originalOriginX, this._originalOriginY);
        this.originX = this._originalOriginX, this.originY = this._originalOriginY, this.left = f.x, this.top = f.y, this._originalOriginX = null, this._originalOriginY = null;
      }, _getLeftTopCoords: function () {
        return this.translateToOriginPoint(this.getCenterPoint(), "left", "top");
      }
    });
  }(), function () {
    var l = v.util, a = l.degreesToRadians, c = l.multiplyTransformMatrices, f = l.transformPoint;
    l.object.extend(v.Object.prototype, {
      oCoords: null, aCoords: null, lineCoords: null, ownMatrixCache: null, matrixCache: null, controls: {}, _getCoords: function (i, r) {
        return r ? i ? this.calcACoords() : this.calcLineCoords() : (this.aCoords && this.lineCoords || this.setCoords(!0), i ? this.aCoords : this.lineCoords);
      }, getCoords: function (i, r) {
        return s = this._getCoords(i, r), [new v.Point(s.tl.x, s.tl.y), new v.Point(s.tr.x, s.tr.y), new v.Point(s.br.x, s.br.y), new v.Point(s.bl.x, s.bl.y)];
        var s;
      }, intersectsWithRect: function (i, r, s, u) {
        var h = this.getCoords(s, u);
        return v.Intersection.intersectPolygonRectangle(h, i, r).status === "Intersection";
      }, intersectsWithObject: function (i, r, s) {
        return v.Intersection.intersectPolygonPolygon(this.getCoords(r, s), i.getCoords(r, s)).status === "Intersection" || i.isContainedWithinObject(this, r, s) || this.isContainedWithinObject(i, r, s);
      }, isContainedWithinObject: function (i, r, s) {
        for (var u = this.getCoords(r, s), h = r ? i.aCoords : i.lineCoords, p = 0, m = i._getImageLines(h); p < 4; p++) if (!i.containsPoint(u[p], m)) return !1;
        return !0;
      }, isContainedWithinRect: function (i, r, s, u) {
        var h = this.getBoundingRect(s, u);
        return h.left >= i.x && h.left + h.width <= r.x && h.top >= i.y && h.top + h.height <= r.y;
      }, containsPoint: function (i, r, s, u) {
        var h = this._getCoords(s, u), p = (r = r || this._getImageLines(h), this._findCrossPoints(i, r));
        return p !== 0 && p % 2 == 1;
      }, isOnScreen: function (i) {
        if (!this.canvas) return !1;
        var r = this.canvas.vptCoords.tl, s = this.canvas.vptCoords.br;
        return !!this.getCoords(!0, i).some(function (u) {
          return u.x <= s.x && u.x >= r.x && u.y <= s.y && u.y >= r.y;
        }) || !!this.intersectsWithRect(r, s, !0, i) || this._containsCenterOfCanvas(r, s, i);
      }, _containsCenterOfCanvas: function (i, r, s) {
        var u = { x: (i.x + r.x) / 2, y: (i.y + r.y) / 2 };
        return !!this.containsPoint(u, null, !0, s);
      }, isPartiallyOnScreen: function (i) {
        if (!this.canvas) return !1;
        var r = this.canvas.vptCoords.tl, s = this.canvas.vptCoords.br;
        return !!this.intersectsWithRect(r, s, !0, i) || this.getCoords(!0, i).every(function (u) {
          return (u.x >= s.x || u.x <= r.x) && (u.y >= s.y || u.y <= r.y);
        }) && this._containsCenterOfCanvas(r, s, i);
      }, _getImageLines: function (i) {
        return { topline: { o: i.tl, d: i.tr }, rightline: { o: i.tr, d: i.br }, bottomline: { o: i.br, d: i.bl }, leftline: { o: i.bl, d: i.tl } };
      }, _findCrossPoints: function (i, r) {
        var s, u, h, p = 0;
        for (var m in r) if (!((h = r[m]).o.y < i.y && h.d.y < i.y || h.o.y >= i.y && h.d.y >= i.y || (h.o.x === h.d.x && h.o.x >= i.x ? u = h.o.x : (s = (h.d.y - h.o.y) / (h.d.x - h.o.x), u = -(i.y - 0 * i.x - (h.o.y - s * h.o.x)) / (0 - s)), u >= i.x && (p += 1), p !== 2))) break;
        return p;
      }, getBoundingRect: function (i, r) {
        var s = this.getCoords(i, r);
        return l.makeBoundingBoxFromPoints(s);
      }, getScaledWidth: function () {
        return this._getTransformedDimensions().x;
      }, getScaledHeight: function () {
        return this._getTransformedDimensions().y;
      }, _constrainScale: function (i) {
        return Math.abs(i) < this.minScaleLimit ? i < 0 ? -this.minScaleLimit : this.minScaleLimit : i === 0 ? 1e-4 : i;
      }, scale: function (i) {
        return this._set("scaleX", i), this._set("scaleY", i), this.setCoords();
      }, scaleToWidth: function (i, r) {
        var s = this.getBoundingRect(r).width / this.getScaledWidth();
        return this.scale(i / this.width / s);
      }, scaleToHeight: function (i, r) {
        var s = this.getBoundingRect(r).height / this.getScaledHeight();
        return this.scale(i / this.height / s);
      }, calcCoords: function (i) {
        return i ? this.calcACoords() : this.calcOCoords();
      }, calcLineCoords: function () {
        var i = this.getViewportTransform(), r = this.padding, s = a(this.angle), u = l.cos(s) * r, h = l.sin(s) * r, p = u + h, m = u - h, y = this.calcACoords(), x = { tl: f(y.tl, i), tr: f(y.tr, i), bl: f(y.bl, i), br: f(y.br, i) };
        return r && (x.tl.x -= m, x.tl.y -= p, x.tr.x += p, x.tr.y -= m, x.bl.x -= p, x.bl.y += m, x.br.x += m, x.br.y += p), x;
      }, calcOCoords: function () {
        var i = this._calcRotateMatrix(), r = this._calcTranslateMatrix(), s = this.getViewportTransform(), u = c(s, r), h = c(u, i), p = (h = c(h, [1 / s[0], 0, 0, 1 / s[3], 0, 0]), this._calculateCurrentDimensions()), m = {};
        return this.forEachControl(function (y, x, S) {
          m[x] = y.positionHandler(p, h, S);
        }), m;
      }, calcACoords: function () {
        var i = this._calcRotateMatrix(), r = this._calcTranslateMatrix(), s = c(r, i), u = this._getTransformedDimensions(), h = u.x / 2, p = u.y / 2;
        return { tl: f({ x: -h, y: -p }, s), tr: f({ x: h, y: -p }, s), bl: f({ x: -h, y: p }, s), br: f({ x: h, y: p }, s) };
      }, setCoords: function (i) {
        return this.aCoords = this.calcACoords(), this.lineCoords = this.group ? this.aCoords : this.calcLineCoords(), i || (this.oCoords = this.calcOCoords(), this._setCornerCoords && this._setCornerCoords()), this;
      }, _calcRotateMatrix: function () {
        return l.calcRotateMatrix(this);
      }, _calcTranslateMatrix: function () {
        var i = this.getCenterPoint();
        return [1, 0, 0, 1, i.x, i.y];
      }, transformMatrixKey: function (i) {
        var r = "_", s = "";
        return !i && this.group && (s = this.group.transformMatrixKey(i) + r), s + this.top + r + this.left + r + this.scaleX + r + this.scaleY + r + this.skewX + r + this.skewY + r + this.angle + r + this.originX + r + this.originY + r + this.width + r + this.height + r + this.strokeWidth + this.flipX + this.flipY;
      }, calcTransformMatrix: function (i) {
        var r = this.calcOwnMatrix();
        if (i || !this.group) return r;
        var s = this.transformMatrixKey(i), u = this.matrixCache || (this.matrixCache = {});
        return u.key === s ? u.value : (this.group && (r = c(this.group.calcTransformMatrix(!1), r)), u.key = s, u.value = r, r);
      }, calcOwnMatrix: function () {
        var i = this.transformMatrixKey(!0), r = this.ownMatrixCache || (this.ownMatrixCache = {});
        if (r.key === i) return r.value;
        var s = this._calcTranslateMatrix(), u = { angle: this.angle, translateX: s[4], translateY: s[5], scaleX: this.scaleX, scaleY: this.scaleY, skewX: this.skewX, skewY: this.skewY, flipX: this.flipX, flipY: this.flipY };
        return r.key = i, r.value = l.composeMatrix(u), r.value;
      }, _calcDimensionsTransformMatrix: function (i, r, s) {
        return l.calcDimensionsMatrix({ skewX: i, skewY: r, scaleX: this.scaleX * (s && this.flipX ? -1 : 1), scaleY: this.scaleY * (s && this.flipY ? -1 : 1) });
      }, _getNonTransformedDimensions: function () {
        var i = this.strokeWidth;
        return { x: this.width + i, y: this.height + i };
      }, _getTransformedDimensions: function (i, r) {
        i === void 0 && (i = this.skewX), r === void 0 && (r = this.skewY);
        var s, u, h, p = i === 0 && r === 0;
        if (this.strokeUniform ? (u = this.width, h = this.height) : (u = (s = this._getNonTransformedDimensions()).x, h = s.y), p) return this._finalizeDimensions(u * this.scaleX, h * this.scaleY);
        var m = l.sizeAfterTransform(u, h, { scaleX: this.scaleX, scaleY: this.scaleY, skewX: i, skewY: r });
        return this._finalizeDimensions(m.x, m.y);
      }, _finalizeDimensions: function (i, r) {
        return this.strokeUniform ? { x: i + this.strokeWidth, y: r + this.strokeWidth } : { x: i, y: r };
      }, _calculateCurrentDimensions: function () {
        var i = this.getViewportTransform(), r = this._getTransformedDimensions();
        return f(r, i, !0).scalarAdd(2 * this.padding);
      }
    });
  }(), v.util.object.extend(v.Object.prototype, {
    sendToBack: function () {
      return this.group ? v.StaticCanvas.prototype.sendToBack.call(this.group, this) : this.canvas && this.canvas.sendToBack(this), this;
    }, bringToFront: function () {
      return this.group ? v.StaticCanvas.prototype.bringToFront.call(this.group, this) : this.canvas && this.canvas.bringToFront(this), this;
    }, sendBackwards: function (l) {
      return this.group ? v.StaticCanvas.prototype.sendBackwards.call(this.group, this, l) : this.canvas && this.canvas.sendBackwards(this, l), this;
    }, bringForward: function (l) {
      return this.group ? v.StaticCanvas.prototype.bringForward.call(this.group, this, l) : this.canvas && this.canvas.bringForward(this, l), this;
    }, moveTo: function (l) {
      return this.group && this.group.type !== "activeSelection" ? v.StaticCanvas.prototype.moveTo.call(this.group, this, l) : this.canvas && this.canvas.moveTo(this, l), this;
    }
  }), function () {
    function l(c, f) {
      if (f) {
        if (f.toLive) return c + ": url(#SVGID_" + f.id + "); ";
        var i = new v.Color(f), r = c + ": " + i.toRgb() + "; ", s = i.getAlpha();
        return s !== 1 && (r += c + "-opacity: " + s.toString() + "; "), r;
      }
      return c + ": none; ";
    }
    var a = v.util.toFixed;
    v.util.object.extend(v.Object.prototype, {
      getSvgStyles: function (c) {
        var f = this.fillRule ? this.fillRule : "nonzero", i = this.strokeWidth ? this.strokeWidth : "0", r = this.strokeDashArray ? this.strokeDashArray.join(" ") : "none", s = this.strokeDashOffset ? this.strokeDashOffset : "0", u = this.strokeLineCap ? this.strokeLineCap : "butt", h = this.strokeLineJoin ? this.strokeLineJoin : "miter", p = this.strokeMiterLimit ? this.strokeMiterLimit : "4", m = this.opacity !== void 0 ? this.opacity : "1", y = this.visible ? "" : " visibility: hidden;", x = c ? "" : this.getSvgFilter(), S = l("fill", this.fill);
        return [l("stroke", this.stroke), "stroke-width: ", i, "; ", "stroke-dasharray: ", r, "; ", "stroke-linecap: ", u, "; ", "stroke-dashoffset: ", s, "; ", "stroke-linejoin: ", h, "; ", "stroke-miterlimit: ", p, "; ", S, "fill-rule: ", f, "; ", "opacity: ", m, ";", x, y].join("");
      }, getSvgSpanStyles: function (c, f) {
        var i = "; ", r = c.fontFamily ? "font-family: " + (c.fontFamily.indexOf("'") === -1 && c.fontFamily.indexOf('"') === -1 ? "'" + c.fontFamily + "'" : c.fontFamily) + i : "", s = c.strokeWidth ? "stroke-width: " + c.strokeWidth + i : "", u = (r = r, c.fontSize ? "font-size: " + c.fontSize + "px" + i : ""), h = c.fontStyle ? "font-style: " + c.fontStyle + i : "", p = c.fontWeight ? "font-weight: " + c.fontWeight + i : "", m = c.fill ? l("fill", c.fill) : "", y = c.stroke ? l("stroke", c.stroke) : "", x = this.getSvgTextDecoration(c);
        return x && (x = "text-decoration: " + x + i), [y, s, r, u, h, p, x, m, c.deltaY ? "baseline-shift: " + -c.deltaY + "; " : "", f ? "white-space: pre; " : ""].join("");
      }, getSvgTextDecoration: function (c) {
        return ["overline", "underline", "line-through"].filter(function (f) {
          return c[f.replace("-", "")];
        }).join(" ");
      }, getSvgFilter: function () {
        return this.shadow ? "filter: url(#SVGID_" + this.shadow.id + ");" : "";
      }, getSvgCommons: function () {
        return [this.id ? 'id="' + this.id + '" ' : "", this.clipPath ? 'clip-path="url(#' + this.clipPath.clipPathId + ')" ' : ""].join("");
      }, getSvgTransform: function (c, f) {
        var i = c ? this.calcTransformMatrix() : this.calcOwnMatrix();
        return 'transform="' + v.util.matrixToSVG(i) + (f || "") + '" ';
      }, _setSVGBg: function (c) {
        if (this.backgroundColor) {
          var f = v.Object.NUM_FRACTION_DIGITS;
          c.push("        <rect ", this._getFillAttributes(this.backgroundColor), ' x="', a(-this.width / 2, f), '" y="', a(-this.height / 2, f), '" width="', a(this.width, f), '" height="', a(this.height, f), `"></rect>
`);
        }
      }, toSVG: function (c) {
        return this._createBaseSVGMarkup(this._toSVG(c), { reviver: c });
      }, toClipPathSVG: function (c) {
        return "    " + this._createBaseClipPathSVGMarkup(this._toSVG(c), { reviver: c });
      }, _createBaseClipPathSVGMarkup: function (c, f) {
        var i = (f = f || {}).reviver, r = f.additionalTransform || "", s = [this.getSvgTransform(!0, r), this.getSvgCommons()].join(""), u = c.indexOf("COMMON_PARTS");
        return c[u] = s, i ? i(c.join("")) : c.join("");
      }, _createBaseSVGMarkup: function (c, f) {
        var i, r, s = (f = f || {}).noStyle, u = f.reviver, h = s ? "" : 'style="' + this.getSvgStyles() + '" ', p = f.withShadow ? 'style="' + this.getSvgFilter() + '" ' : "", m = this.clipPath, y = this.strokeUniform ? 'vector-effect="non-scaling-stroke" ' : "", x = m && m.absolutePositioned, S = this.stroke, k = this.fill, D = this.shadow, M = [], C = c.indexOf("COMMON_PARTS"), O = f.additionalTransform;
        return m && (m.clipPathId = "CLIPPATH_" + v.Object.__uid++, r = '<clipPath id="' + m.clipPathId + `" >
` + m.toClipPathSVG(u) + `</clipPath>
`), x && M.push("<g ", p, this.getSvgCommons(), ` >
`), M.push("<g ", this.getSvgTransform(!1), x ? "" : p + this.getSvgCommons(), ` >
`), i = [h, y, s ? "" : this.addPaintOrder(), " ", O ? 'transform="' + O + '" ' : ""].join(""), c[C] = i, k && k.toLive && M.push(k.toSVG(this)), S && S.toLive && M.push(S.toSVG(this)), D && M.push(D.toSVG(this)), m && M.push(r), M.push(c.join("")), M.push(`</g>
`), x && M.push(`</g>
`), u ? u(M.join("")) : M.join("");
      }, addPaintOrder: function () {
        return this.paintFirst !== "fill" ? ' paint-order="' + this.paintFirst + '" ' : "";
      }
    });
  }(), function () {
    var l = v.util.object.extend, a = "stateProperties";
    function c(i, r, s) {
      var u = {};
      s.forEach(function (h) {
        u[h] = i[h];
      }), l(i[r], u, !0);
    }
    function f(i, r, s) {
      if (i === r) return !0;
      if (Array.isArray(i)) {
        if (!Array.isArray(r) || i.length !== r.length) return !1;
        for (var u = 0, h = i.length; u < h; u++) if (!f(i[u], r[u])) return !1;
        return !0;
      }
      if (i && typeof i == "object") {
        var p, m = Object.keys(i);
        if (!r || typeof r != "object" || !s && m.length !== Object.keys(r).length) return !1;
        for (u = 0, h = m.length; u < h; u++) if ((p = m[u]) !== "canvas" && p !== "group" && !f(i[p], r[p])) return !1;
        return !0;
      }
    }
    v.util.object.extend(v.Object.prototype, {
      hasStateChanged: function (i) {
        var r = "_" + (i = i || a);
        return Object.keys(this[r]).length < this[i].length || !f(this[r], this, !0);
      }, saveState: function (i) {
        var r = i && i.propertySet || a, s = "_" + r;
        return this[s] ? (c(this, s, this[r]), i && i.stateProperties && c(this, s, i.stateProperties), this) : this.setupState(i);
      }, setupState: function (i) {
        var r = (i = i || {}).propertySet || a;
        return i.propertySet = r, this["_" + r] = {}, this.saveState(i), this;
      }
    });
  }(), function () {
    var l = v.util.degreesToRadians;
    v.util.object.extend(v.Object.prototype, {
      _findTargetCorner: function (a, c) {
        if (!this.hasControls || this.group || !this.canvas || this.canvas._activeObject !== this) return !1;
        var f, i, r, s = a.x, u = a.y, h = Object.keys(this.oCoords), p = h.length - 1;
        for (this.__corner = 0; p >= 0; p--) if (r = h[p], this.isControlVisible(r) && (i = this._getImageLines(c ? this.oCoords[r].touchCorner : this.oCoords[r].corner), (f = this._findCrossPoints({ x: s, y: u }, i)) !== 0 && f % 2 == 1)) return this.__corner = r, r;
        return !1;
      }, forEachControl: function (a) {
        for (var c in this.controls) a(this.controls[c], c, this);
      }, _setCornerCoords: function () {
        var a = this.oCoords;
        for (var c in a) {
          var f = this.controls[c];
          a[c].corner = f.calcCornerCoords(this.angle, this.cornerSize, a[c].x, a[c].y, !1), a[c].touchCorner = f.calcCornerCoords(this.angle, this.touchCornerSize, a[c].x, a[c].y, !0);
        }
      }, drawSelectionBackground: function (a) {
        if (!this.selectionBackgroundColor || this.canvas && !this.canvas.interactive || this.canvas && this.canvas._activeObject !== this) return this;
        a.save();
        var c = this.getCenterPoint(), f = this._calculateCurrentDimensions(), i = this.canvas.viewportTransform;
        return a.translate(c.x, c.y), a.scale(1 / i[0], 1 / i[3]), a.rotate(l(this.angle)), a.fillStyle = this.selectionBackgroundColor, a.fillRect(-f.x / 2, -f.y / 2, f.x, f.y), a.restore(), this;
      }, drawBorders: function (a, c) {
        c = c || {};
        var f = this._calculateCurrentDimensions(), i = this.borderScaleFactor, r = f.x + i, s = f.y + i, u = c.hasControls !== void 0 ? c.hasControls : this.hasControls, h = !1;
        return a.save(), a.strokeStyle = c.borderColor || this.borderColor, this._setLineDash(a, c.borderDashArray || this.borderDashArray), a.strokeRect(-r / 2, -s / 2, r, s), u && (a.beginPath(), this.forEachControl(function (p, m, y) {
          p.withConnection && p.getVisibility(y, m) && (h = !0, a.moveTo(p.x * r, p.y * s), a.lineTo(p.x * r + p.offsetX, p.y * s + p.offsetY));
        }), h && a.stroke()), a.restore(), this;
      }, drawBordersInGroup: function (a, c, f) {
        f = f || {};
        var i = v.util.sizeAfterTransform(this.width, this.height, c), r = this.strokeWidth, s = this.strokeUniform, u = this.borderScaleFactor, h = i.x + r * (s ? this.canvas.getZoom() : c.scaleX) + u, p = i.y + r * (s ? this.canvas.getZoom() : c.scaleY) + u;
        return a.save(), this._setLineDash(a, f.borderDashArray || this.borderDashArray), a.strokeStyle = f.borderColor || this.borderColor, a.strokeRect(-h / 2, -p / 2, h, p), a.restore(), this;
      }, drawControls: function (a, c) {
        c = c || {}, a.save();
        var f, i, r = this.canvas.getRetinaScaling();
        return a.setTransform(r, 0, 0, r, 0, 0), a.strokeStyle = a.fillStyle = c.cornerColor || this.cornerColor, this.transparentCorners || (a.strokeStyle = c.cornerStrokeColor || this.cornerStrokeColor), this._setLineDash(a, c.cornerDashArray || this.cornerDashArray), this.setCoords(), this.group && (f = this.group.calcTransformMatrix()), this.forEachControl(function (s, u, h) {
          i = h.oCoords[u], s.getVisibility(h, u) && (f && (i = v.util.transformPoint(i, f)), s.render(a, i.x, i.y, c, h));
        }), a.restore(), this;
      }, isControlVisible: function (a) {
        return this.controls[a] && this.controls[a].getVisibility(this, a);
      }, setControlVisible: function (a, c) {
        return this._controlsVisibility || (this._controlsVisibility = {}), this._controlsVisibility[a] = c, this;
      }, setControlsVisibility: function (a) {
        for (var c in a || (a = {}), a) this.setControlVisible(c, a[c]);
        return this;
      }, onDeselect: function () {
      }, onSelect: function () {
      }
    });
  }(), v.util.object.extend(v.StaticCanvas.prototype, {
    FX_DURATION: 500, fxCenterObjectH: function (l, a) {
      var c = function () {
      }, f = (a = a || {}).onComplete || c, i = a.onChange || c, r = this;
      return v.util.animate({
        startValue: l.left, endValue: this.getCenter().left, duration: this.FX_DURATION, onChange: function (s) {
          l.set("left", s), r.requestRenderAll(), i();
        }, onComplete: function () {
          l.setCoords(), f();
        }
      }), this;
    }, fxCenterObjectV: function (l, a) {
      var c = function () {
      }, f = (a = a || {}).onComplete || c, i = a.onChange || c, r = this;
      return v.util.animate({
        startValue: l.top, endValue: this.getCenter().top, duration: this.FX_DURATION, onChange: function (s) {
          l.set("top", s), r.requestRenderAll(), i();
        }, onComplete: function () {
          l.setCoords(), f();
        }
      }), this;
    }, fxRemove: function (l, a) {
      var c = function () {
      }, f = (a = a || {}).onComplete || c, i = a.onChange || c, r = this;
      return v.util.animate({
        startValue: l.opacity, endValue: 0, duration: this.FX_DURATION, onChange: function (s) {
          l.set("opacity", s), r.requestRenderAll(), i();
        }, onComplete: function () {
          r.remove(l), f();
        }
      }), this;
    }
  }), v.util.object.extend(v.Object.prototype, {
    animate: function () {
      if (arguments[0] && typeof arguments[0] == "object") {
        var l, a, c = [];
        for (l in arguments[0]) c.push(l);
        for (var f = 0, i = c.length; f < i; f++) l = c[f], a = f !== i - 1, this._animate(l, arguments[0][l], arguments[1], a);
      } else this._animate.apply(this, arguments);
      return this;
    }, _animate: function (l, a, c, f) {
      var i, r = this;
      a = a.toString(), c = c ? v.util.object.clone(c) : {}, ~l.indexOf(".") && (i = l.split("."));
      var s = r.colorProperties.indexOf(l) > -1 || i && r.colorProperties.indexOf(i[1]) > -1, u = i ? this.get(i[0])[i[1]] : this.get(l);
      "from" in c || (c.from = u), s || (a = ~a.indexOf("=") ? u + parseFloat(a.replace("=", "")) : parseFloat(a));
      var h = {
        startValue: c.from, endValue: a, byValue: c.by, easing: c.easing, duration: c.duration, abort: c.abort && function (p, m, y) {
          return c.abort.call(r, p, m, y);
        }, onChange: function (p, m, y) {
          i ? r[i[0]][i[1]] = p : r.set(l, p), f || c.onChange && c.onChange(p, m, y);
        }, onComplete: function (p, m, y) {
          f || (r.setCoords(), c.onComplete && c.onComplete(p, m, y));
        }
      };
      return s ? v.util.animateColor(h.startValue, h.endValue, h.duration, h) : v.util.animate(h);
    }
  }), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.object.extend, f = a.util.object.clone, i = { x1: 1, x2: 1, y1: 1, y2: 1 };
    function r(s, u) {
      var h = s.origin, p = s.axis1, m = s.axis2, y = s.dimension, x = u.nearest, S = u.center, k = u.farthest;
      return function () {
        switch (this.get(h)) {
          case x:
            return Math.min(this.get(p), this.get(m));
          case S:
            return Math.min(this.get(p), this.get(m)) + 0.5 * this.get(y);
          case k:
            return Math.max(this.get(p), this.get(m));
        }
      };
    }
    a.Line ? a.warn("fabric.Line is already defined") : (a.Line = a.util.createClass(a.Object, {
      type: "line", x1: 0, y1: 0, x2: 0, y2: 0, cacheProperties: a.Object.prototype.cacheProperties.concat("x1", "x2", "y1", "y2"), initialize: function (s, u) {
        s || (s = [0, 0, 0, 0]), this.callSuper("initialize", u), this.set("x1", s[0]), this.set("y1", s[1]), this.set("x2", s[2]), this.set("y2", s[3]), this._setWidthHeight(u);
      }, _setWidthHeight: function (s) {
        s || (s = {}), this.width = Math.abs(this.x2 - this.x1), this.height = Math.abs(this.y2 - this.y1), this.left = "left" in s ? s.left : this._getLeftToOriginX(), this.top = "top" in s ? s.top : this._getTopToOriginY();
      }, _set: function (s, u) {
        return this.callSuper("_set", s, u), i[s] !== void 0 && this._setWidthHeight(), this;
      }, _getLeftToOriginX: r({ origin: "originX", axis1: "x1", axis2: "x2", dimension: "width" }, { nearest: "left", center: "center", farthest: "right" }), _getTopToOriginY: r({ origin: "originY", axis1: "y1", axis2: "y2", dimension: "height" }, { nearest: "top", center: "center", farthest: "bottom" }), _render: function (s) {
        s.beginPath();
        var u = this.calcLinePoints();
        s.moveTo(u.x1, u.y1), s.lineTo(u.x2, u.y2), s.lineWidth = this.strokeWidth;
        var h = s.strokeStyle;
        s.strokeStyle = this.stroke || s.fillStyle, this.stroke && this._renderStroke(s), s.strokeStyle = h;
      }, _findCenterFromElement: function () {
        return { x: (this.x1 + this.x2) / 2, y: (this.y1 + this.y2) / 2 };
      }, toObject: function (s) {
        return c(this.callSuper("toObject", s), this.calcLinePoints());
      }, _getNonTransformedDimensions: function () {
        var s = this.callSuper("_getNonTransformedDimensions");
        return this.strokeLineCap === "butt" && (this.width === 0 && (s.y -= this.strokeWidth), this.height === 0 && (s.x -= this.strokeWidth)), s;
      }, calcLinePoints: function () {
        var s = this.x1 <= this.x2 ? -1 : 1, u = this.y1 <= this.y2 ? -1 : 1, h = s * this.width * 0.5, p = u * this.height * 0.5;
        return { x1: h, x2: s * this.width * -0.5, y1: p, y2: u * this.height * -0.5 };
      }, _toSVG: function () {
        var s = this.calcLinePoints();
        return ["<line ", "COMMON_PARTS", 'x1="', s.x1, '" y1="', s.y1, '" x2="', s.x2, '" y2="', s.y2, `" />
`];
      }
    }), a.Line.ATTRIBUTE_NAMES = a.SHARED_ATTRIBUTES.concat("x1 y1 x2 y2".split(" ")), a.Line.fromElement = function (s, u, h) {
      h = h || {};
      var p = a.parseAttributes(s, a.Line.ATTRIBUTE_NAMES), m = [p.x1 || 0, p.y1 || 0, p.x2 || 0, p.y2 || 0];
      u(new a.Line(m, c(p, h)));
    }, a.Line.fromObject = function (s, u) {
      var h = f(s, !0);
      h.points = [s.x1, s.y1, s.x2, s.y2], a.Object._fromObject("Line", h, function (p) {
        delete p.points, u && u(p);
      }, "points");
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = Math.PI;
    a.Circle ? a.warn("fabric.Circle is already defined.") : (a.Circle = a.util.createClass(a.Object, {
      type: "circle", radius: 0, startAngle: 0, endAngle: 2 * c, cacheProperties: a.Object.prototype.cacheProperties.concat("radius", "startAngle", "endAngle"), _set: function (f, i) {
        return this.callSuper("_set", f, i), f === "radius" && this.setRadius(i), this;
      }, toObject: function (f) {
        return this.callSuper("toObject", ["radius", "startAngle", "endAngle"].concat(f));
      }, _toSVG: function () {
        var f, i = (this.endAngle - this.startAngle) % (2 * c);
        if (i === 0) f = ["<circle ", "COMMON_PARTS", 'cx="0" cy="0" ', 'r="', this.radius, `" />
`];
        else {
          var r = a.util.cos(this.startAngle) * this.radius, s = a.util.sin(this.startAngle) * this.radius, u = a.util.cos(this.endAngle) * this.radius, h = a.util.sin(this.endAngle) * this.radius, p = i > c ? "1" : "0";
          f = ['<path d="M ' + r + " " + s, " A " + this.radius + " " + this.radius, " 0 ", +p + " 1", " " + u + " " + h, '" ', "COMMON_PARTS", ` />
`];
        }
        return f;
      }, _render: function (f) {
        f.beginPath(), f.arc(0, 0, this.radius, this.startAngle, this.endAngle, !1), this._renderPaintInOrder(f);
      }, getRadiusX: function () {
        return this.get("radius") * this.get("scaleX");
      }, getRadiusY: function () {
        return this.get("radius") * this.get("scaleY");
      }, setRadius: function (f) {
        return this.radius = f, this.set("width", 2 * f).set("height", 2 * f);
      }
    }), a.Circle.ATTRIBUTE_NAMES = a.SHARED_ATTRIBUTES.concat("cx cy r".split(" ")), a.Circle.fromElement = function (f, i) {
      var r, s = a.parseAttributes(f, a.Circle.ATTRIBUTE_NAMES);
      if (!("radius" in (r = s) && r.radius >= 0)) throw new Error("value of `r` attribute is required and can not be negative");
      s.left = (s.left || 0) - s.radius, s.top = (s.top || 0) - s.radius, i(new a.Circle(s));
    }, a.Circle.fromObject = function (f, i) {
      a.Object._fromObject("Circle", f, i);
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {});
    a.Triangle ? a.warn("fabric.Triangle is already defined") : (a.Triangle = a.util.createClass(a.Object, {
      type: "triangle", width: 100, height: 100, _render: function (c) {
        var f = this.width / 2, i = this.height / 2;
        c.beginPath(), c.moveTo(-f, i), c.lineTo(0, -i), c.lineTo(f, i), c.closePath(), this._renderPaintInOrder(c);
      }, _toSVG: function () {
        var c = this.width / 2, f = this.height / 2;
        return ["<polygon ", "COMMON_PARTS", 'points="', [-c + " " + f, "0 " + -f, c + " " + f].join(","), '" />'];
      }
    }), a.Triangle.fromObject = function (c, f) {
      return a.Object._fromObject("Triangle", c, f);
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = 2 * Math.PI;
    a.Ellipse ? a.warn("fabric.Ellipse is already defined.") : (a.Ellipse = a.util.createClass(a.Object, {
      type: "ellipse", rx: 0, ry: 0, cacheProperties: a.Object.prototype.cacheProperties.concat("rx", "ry"), initialize: function (f) {
        this.callSuper("initialize", f), this.set("rx", f && f.rx || 0), this.set("ry", f && f.ry || 0);
      }, _set: function (f, i) {
        switch (this.callSuper("_set", f, i), f) {
          case "rx":
            this.rx = i, this.set("width", 2 * i);
            break;
          case "ry":
            this.ry = i, this.set("height", 2 * i);
        }
        return this;
      }, getRx: function () {
        return this.get("rx") * this.get("scaleX");
      }, getRy: function () {
        return this.get("ry") * this.get("scaleY");
      }, toObject: function (f) {
        return this.callSuper("toObject", ["rx", "ry"].concat(f));
      }, _toSVG: function () {
        return ["<ellipse ", "COMMON_PARTS", 'cx="0" cy="0" ', 'rx="', this.rx, '" ry="', this.ry, `" />
`];
      }, _render: function (f) {
        f.beginPath(), f.save(), f.transform(1, 0, 0, this.ry / this.rx, 0, 0), f.arc(0, 0, this.rx, 0, c, !1), f.restore(), this._renderPaintInOrder(f);
      }
    }), a.Ellipse.ATTRIBUTE_NAMES = a.SHARED_ATTRIBUTES.concat("cx cy rx ry".split(" ")), a.Ellipse.fromElement = function (f, i) {
      var r = a.parseAttributes(f, a.Ellipse.ATTRIBUTE_NAMES);
      r.left = (r.left || 0) - r.rx, r.top = (r.top || 0) - r.ry, i(new a.Ellipse(r));
    }, a.Ellipse.fromObject = function (f, i) {
      a.Object._fromObject("Ellipse", f, i);
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.object.extend;
    a.Rect ? a.warn("fabric.Rect is already defined") : (a.Rect = a.util.createClass(a.Object, {
      stateProperties: a.Object.prototype.stateProperties.concat("rx", "ry"), type: "rect", rx: 0, ry: 0, cacheProperties: a.Object.prototype.cacheProperties.concat("rx", "ry"), initialize: function (f) {
        this.callSuper("initialize", f), this._initRxRy();
      }, _initRxRy: function () {
        this.rx && !this.ry ? this.ry = this.rx : this.ry && !this.rx && (this.rx = this.ry);
      }, _render: function (f) {
        var i = this.rx ? Math.min(this.rx, this.width / 2) : 0, r = this.ry ? Math.min(this.ry, this.height / 2) : 0, s = this.width, u = this.height, h = -this.width / 2, p = -this.height / 2, m = i !== 0 || r !== 0, y = 0.4477152502;
        f.beginPath(), f.moveTo(h + i, p), f.lineTo(h + s - i, p), m && f.bezierCurveTo(h + s - y * i, p, h + s, p + y * r, h + s, p + r), f.lineTo(h + s, p + u - r), m && f.bezierCurveTo(h + s, p + u - y * r, h + s - y * i, p + u, h + s - i, p + u), f.lineTo(h + i, p + u), m && f.bezierCurveTo(h + y * i, p + u, h, p + u - y * r, h, p + u - r), f.lineTo(h, p + r), m && f.bezierCurveTo(h, p + y * r, h + y * i, p, h + i, p), f.closePath(), this._renderPaintInOrder(f);
      }, toObject: function (f) {
        return this.callSuper("toObject", ["rx", "ry"].concat(f));
      }, _toSVG: function () {
        return ["<rect ", "COMMON_PARTS", 'x="', -this.width / 2, '" y="', -this.height / 2, '" rx="', this.rx, '" ry="', this.ry, '" width="', this.width, '" height="', this.height, `" />
`];
      }
    }), a.Rect.ATTRIBUTE_NAMES = a.SHARED_ATTRIBUTES.concat("x y rx ry width height".split(" ")), a.Rect.fromElement = function (f, i, r) {
      if (!f) return i(null);
      r = r || {};
      var s = a.parseAttributes(f, a.Rect.ATTRIBUTE_NAMES);
      s.left = s.left || 0, s.top = s.top || 0, s.height = s.height || 0, s.width = s.width || 0;
      var u = new a.Rect(c(r ? a.util.object.clone(r) : {}, s));
      u.visible = u.visible && u.width > 0 && u.height > 0, i(u);
    }, a.Rect.fromObject = function (f, i) {
      return a.Object._fromObject("Rect", f, i);
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.object.extend, f = a.util.array.min, i = a.util.array.max, r = a.util.toFixed;
    a.Polyline ? a.warn("fabric.Polyline is already defined") : (a.Polyline = a.util.createClass(a.Object, {
      type: "polyline", points: null, cacheProperties: a.Object.prototype.cacheProperties.concat("points"), initialize: function (s, u) {
        u = u || {}, this.points = s || [], this.callSuper("initialize", u), this._setPositionDimensions(u);
      }, _setPositionDimensions: function (s) {
        var u, h = this._calcDimensions(s);
        this.width = h.width, this.height = h.height, s.fromSVG || (u = this.translateToGivenOrigin({ x: h.left - this.strokeWidth / 2, y: h.top - this.strokeWidth / 2 }, "left", "top", this.originX, this.originY)), s.left === void 0 && (this.left = s.fromSVG ? h.left : u.x), s.top === void 0 && (this.top = s.fromSVG ? h.top : u.y), this.pathOffset = { x: h.left + this.width / 2, y: h.top + this.height / 2 };
      }, _calcDimensions: function () {
        var s = this.points, u = f(s, "x") || 0, h = f(s, "y") || 0;
        return { left: u, top: h, width: (i(s, "x") || 0) - u, height: (i(s, "y") || 0) - h };
      }, toObject: function (s) {
        return c(this.callSuper("toObject", s), { points: this.points.concat() });
      }, _toSVG: function () {
        for (var s = [], u = this.pathOffset.x, h = this.pathOffset.y, p = a.Object.NUM_FRACTION_DIGITS, m = 0, y = this.points.length; m < y; m++) s.push(r(this.points[m].x - u, p), ",", r(this.points[m].y - h, p), " ");
        return ["<" + this.type + " ", "COMMON_PARTS", 'points="', s.join(""), `" />
`];
      }, commonRender: function (s) {
        var u, h = this.points.length, p = this.pathOffset.x, m = this.pathOffset.y;
        if (!h || isNaN(this.points[h - 1].y)) return !1;
        s.beginPath(), s.moveTo(this.points[0].x - p, this.points[0].y - m);
        for (var y = 0; y < h; y++) u = this.points[y], s.lineTo(u.x - p, u.y - m);
        return !0;
      }, _render: function (s) {
        this.commonRender(s) && this._renderPaintInOrder(s);
      }, complexity: function () {
        return this.get("points").length;
      }
    }), a.Polyline.ATTRIBUTE_NAMES = a.SHARED_ATTRIBUTES.concat(), a.Polyline.fromElementGenerator = function (s) {
      return function (u, h, p) {
        if (!u) return h(null);
        p || (p = {});
        var m = a.parsePointsAttribute(u.getAttribute("points")), y = a.parseAttributes(u, a[s].ATTRIBUTE_NAMES);
        y.fromSVG = !0, h(new a[s](m, c(y, p)));
      };
    }, a.Polyline.fromElement = a.Polyline.fromElementGenerator("Polyline"), a.Polyline.fromObject = function (s, u) {
      return a.Object._fromObject("Polyline", s, u, "points");
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {});
    a.Polygon ? a.warn("fabric.Polygon is already defined") : (a.Polygon = a.util.createClass(a.Polyline, {
      type: "polygon", _render: function (c) {
        this.commonRender(c) && (c.closePath(), this._renderPaintInOrder(c));
      }
    }), a.Polygon.ATTRIBUTE_NAMES = a.SHARED_ATTRIBUTES.concat(), a.Polygon.fromElement = a.Polyline.fromElementGenerator("Polygon"), a.Polygon.fromObject = function (c, f) {
      a.Object._fromObject("Polygon", c, f, "points");
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.array.min, f = a.util.array.max, i = a.util.object.extend, r = Object.prototype.toString, s = a.util.toFixed;
    a.Path ? a.warn("fabric.Path is already defined") : (a.Path = a.util.createClass(a.Object, {
      type: "path", path: null, cacheProperties: a.Object.prototype.cacheProperties.concat("path", "fillRule"), stateProperties: a.Object.prototype.stateProperties.concat("path"), initialize: function (u, h) {
        h = h || {}, this.callSuper("initialize", h), u || (u = []);
        var p = r.call(u) === "[object Array]";
        this.path = a.util.makePathSimpler(p ? u : a.util.parsePath(u)), this.path && a.Polyline.prototype._setPositionDimensions.call(this, h);
      }, _renderPathCommands: function (u) {
        var h, p = 0, m = 0, y = 0, x = 0, S = 0, k = 0, D = -this.pathOffset.x, M = -this.pathOffset.y;
        u.beginPath();
        for (var C = 0, O = this.path.length; C < O; ++C) switch ((h = this.path[C])[0]) {
          case "L":
            y = h[1], x = h[2], u.lineTo(y + D, x + M);
            break;
          case "M":
            p = y = h[1], m = x = h[2], u.moveTo(y + D, x + M);
            break;
          case "C":
            y = h[5], x = h[6], S = h[3], k = h[4], u.bezierCurveTo(h[1] + D, h[2] + M, S + D, k + M, y + D, x + M);
            break;
          case "Q":
            u.quadraticCurveTo(h[1] + D, h[2] + M, h[3] + D, h[4] + M), y = h[3], x = h[4], S = h[1], k = h[2];
            break;
          case "z":
          case "Z":
            y = p, x = m, u.closePath();
        }
      }, _render: function (u) {
        this._renderPathCommands(u), this._renderPaintInOrder(u);
      }, toString: function () {
        return "#<fabric.Path (" + this.complexity() + '): { "top": ' + this.top + ', "left": ' + this.left + " }>";
      }, toObject: function (u) {
        return i(this.callSuper("toObject", u), {
          path: this.path.map(function (h) {
            return h.slice();
          })
        });
      }, toDatalessObject: function (u) {
        var h = this.toObject(["sourcePath"].concat(u));
        return h.sourcePath && delete h.path, h;
      }, _toSVG: function () {
        return ["<path ", "COMMON_PARTS", 'd="', a.util.joinPath(this.path), '" stroke-linecap="round" ', `/>
`];
      }, _getOffsetTransform: function () {
        var u = a.Object.NUM_FRACTION_DIGITS;
        return " translate(" + s(-this.pathOffset.x, u) + ", " + s(-this.pathOffset.y, u) + ")";
      }, toClipPathSVG: function (u) {
        var h = this._getOffsetTransform();
        return "    " + this._createBaseClipPathSVGMarkup(this._toSVG(), { reviver: u, additionalTransform: h });
      }, toSVG: function (u) {
        var h = this._getOffsetTransform();
        return this._createBaseSVGMarkup(this._toSVG(), { reviver: u, additionalTransform: h });
      }, complexity: function () {
        return this.path.length;
      }, _calcDimensions: function () {
        for (var u, h, p = [], m = [], y = 0, x = 0, S = 0, k = 0, D = 0, M = this.path.length; D < M; ++D) {
          switch ((u = this.path[D])[0]) {
            case "L":
              S = u[1], k = u[2], h = [];
              break;
            case "M":
              y = S = u[1], x = k = u[2], h = [];
              break;
            case "C":
              h = a.util.getBoundsOfCurve(S, k, u[1], u[2], u[3], u[4], u[5], u[6]), S = u[5], k = u[6];
              break;
            case "Q":
              h = a.util.getBoundsOfCurve(S, k, u[1], u[2], u[1], u[2], u[3], u[4]), S = u[3], k = u[4];
              break;
            case "z":
            case "Z":
              S = y, k = x;
          }
          h.forEach(function (N) {
            p.push(N.x), m.push(N.y);
          }), p.push(S), m.push(k);
        }
        var C = c(p) || 0, O = c(m) || 0;
        return { left: C, top: O, width: (f(p) || 0) - C, height: (f(m) || 0) - O };
      }
    }), a.Path.fromObject = function (u, h) {
      if (typeof u.sourcePath == "string") {
        var p = u.sourcePath;
        a.loadSVGFromURL(p, function (m) {
          var y = m[0];
          y.setOptions(u), h && h(y);
        });
      } else a.Object._fromObject("Path", u, h, "path");
    }, a.Path.ATTRIBUTE_NAMES = a.SHARED_ATTRIBUTES.concat(["d"]), a.Path.fromElement = function (u, h, p) {
      var m = a.parseAttributes(u, a.Path.ATTRIBUTE_NAMES);
      m.fromSVG = !0, h(new a.Path(m.d, i(m, p)));
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.array.min, f = a.util.array.max;
    a.Group || (a.Group = a.util.createClass(a.Object, a.Collection, {
      type: "group", strokeWidth: 0, subTargetCheck: !1, cacheProperties: [], useSetOnGroup: !1, initialize: function (i, r, s) {
        r = r || {}, this._objects = [], s && this.callSuper("initialize", r), this._objects = i || [];
        for (var u = this._objects.length; u--;) this._objects[u].group = this;
        if (s) this._updateObjectsACoords();
        else {
          var h = r && r.centerPoint;
          r.originX !== void 0 && (this.originX = r.originX), r.originY !== void 0 && (this.originY = r.originY), h || this._calcBounds(), this._updateObjectsCoords(h), delete r.centerPoint, this.callSuper("initialize", r);
        }
        this.setCoords();
      }, _updateObjectsACoords: function () {
        for (var i = this._objects.length; i--;) this._objects[i].setCoords(!0);
      }, _updateObjectsCoords: function (i) {
        i = i || this.getCenterPoint();
        for (var r = this._objects.length; r--;) this._updateObjectCoords(this._objects[r], i);
      }, _updateObjectCoords: function (i, r) {
        var s = i.left, u = i.top;
        i.set({ left: s - r.x, top: u - r.y }), i.group = this, i.setCoords(!0);
      }, toString: function () {
        return "#<fabric.Group: (" + this.complexity() + ")>";
      }, addWithUpdate: function (i) {
        var r = !!this.group;
        return this._restoreObjectsState(), a.util.resetObjectTransform(this), i && (r && a.util.removeTransformFromObject(i, this.group.calcTransformMatrix()), this._objects.push(i), i.group = this, i._set("canvas", this.canvas)), this._calcBounds(), this._updateObjectsCoords(), this.dirty = !0, r ? this.group.addWithUpdate() : this.setCoords(), this;
      }, removeWithUpdate: function (i) {
        return this._restoreObjectsState(), a.util.resetObjectTransform(this), this.remove(i), this._calcBounds(), this._updateObjectsCoords(), this.setCoords(), this.dirty = !0, this;
      }, _onObjectAdded: function (i) {
        this.dirty = !0, i.group = this, i._set("canvas", this.canvas);
      }, _onObjectRemoved: function (i) {
        this.dirty = !0, delete i.group;
      }, _set: function (i, r) {
        var s = this._objects.length;
        if (this.useSetOnGroup) for (; s--;) this._objects[s].setOnGroup(i, r);
        if (i === "canvas") for (; s--;) this._objects[s]._set(i, r);
        a.Object.prototype._set.call(this, i, r);
      }, toObject: function (i) {
        var r = this.includeDefaultValues, s = this._objects.filter(function (h) {
          return !h.excludeFromExport;
        }).map(function (h) {
          var p = h.includeDefaultValues;
          h.includeDefaultValues = r;
          var m = h.toObject(i);
          return h.includeDefaultValues = p, m;
        }), u = a.Object.prototype.toObject.call(this, i);
        return u.objects = s, u;
      }, toDatalessObject: function (i) {
        var r, s = this.sourcePath;
        if (s) r = s;
        else {
          var u = this.includeDefaultValues;
          r = this._objects.map(function (p) {
            var m = p.includeDefaultValues;
            p.includeDefaultValues = u;
            var y = p.toDatalessObject(i);
            return p.includeDefaultValues = m, y;
          });
        }
        var h = a.Object.prototype.toDatalessObject.call(this, i);
        return h.objects = r, h;
      }, render: function (i) {
        this._transformDone = !0, this.callSuper("render", i), this._transformDone = !1;
      }, shouldCache: function () {
        var i = a.Object.prototype.shouldCache.call(this);
        if (i) {
          for (var r = 0, s = this._objects.length; r < s; r++) if (this._objects[r].willDrawShadow()) return this.ownCaching = !1, !1;
        }
        return i;
      }, willDrawShadow: function () {
        if (a.Object.prototype.willDrawShadow.call(this)) return !0;
        for (var i = 0, r = this._objects.length; i < r; i++) if (this._objects[i].willDrawShadow()) return !0;
        return !1;
      }, isOnACache: function () {
        return this.ownCaching || this.group && this.group.isOnACache();
      }, drawObject: function (i) {
        for (var r = 0, s = this._objects.length; r < s; r++) this._objects[r].render(i);
        this._drawClipPath(i);
      }, isCacheDirty: function (i) {
        if (this.callSuper("isCacheDirty", i)) return !0;
        if (!this.statefullCache) return !1;
        for (var r = 0, s = this._objects.length; r < s; r++) if (this._objects[r].isCacheDirty(!0)) {
          if (this._cacheCanvas) {
            var u = this.cacheWidth / this.zoomX, h = this.cacheHeight / this.zoomY;
            this._cacheContext.clearRect(-u / 2, -h / 2, u, h);
          }
          return !0;
        }
        return !1;
      }, _restoreObjectsState: function () {
        var i = this.calcOwnMatrix();
        return this._objects.forEach(function (r) {
          a.util.addTransformToObject(r, i), delete r.group, r.setCoords();
        }), this;
      }, realizeTransform: function (i, r) {
        return a.util.addTransformToObject(i, r), i;
      }, destroy: function () {
        return this._objects.forEach(function (i) {
          i.set("dirty", !0);
        }), this._restoreObjectsState();
      }, toActiveSelection: function () {
        if (this.canvas) {
          var i = this._objects, r = this.canvas;
          this._objects = [];
          var s = this.toObject();
          delete s.objects;
          var u = new a.ActiveSelection([]);
          return u.set(s), u.type = "activeSelection", r.remove(this), i.forEach(function (h) {
            h.group = u, h.dirty = !0, r.add(h);
          }), u.canvas = r, u._objects = i, r._activeObject = u, u.setCoords(), u;
        }
      }, ungroupOnCanvas: function () {
        return this._restoreObjectsState();
      }, setObjectsCoords: function () {
        return this.forEachObject(function (i) {
          i.setCoords(!0);
        }), this;
      }, _calcBounds: function (i) {
        for (var r, s, u, h, p = [], m = [], y = ["tr", "br", "bl", "tl"], x = 0, S = this._objects.length, k = y.length; x < S; ++x) {
          for (u = (r = this._objects[x]).calcACoords(), h = 0; h < k; h++) s = y[h], p.push(u[s].x), m.push(u[s].y);
          r.aCoords = u;
        }
        this._getBounds(p, m, i);
      }, _getBounds: function (i, r, s) {
        var u = new a.Point(c(i), c(r)), h = new a.Point(f(i), f(r)), p = u.y || 0, m = u.x || 0, y = h.x - u.x || 0, x = h.y - u.y || 0;
        this.width = y, this.height = x, s || this.setPositionByOrigin({ x: m, y: p }, "left", "top");
      }, _toSVG: function (i) {
        for (var r = ["<g ", "COMMON_PARTS", ` >
`], s = 0, u = this._objects.length; s < u; s++) r.push("        ", this._objects[s].toSVG(i));
        return r.push(`</g>
`), r;
      }, getSvgStyles: function () {
        var i = this.opacity !== void 0 && this.opacity !== 1 ? "opacity: " + this.opacity + ";" : "", r = this.visible ? "" : " visibility: hidden;";
        return [i, this.getSvgFilter(), r].join("");
      }, toClipPathSVG: function (i) {
        for (var r = [], s = 0, u = this._objects.length; s < u; s++) r.push("    ", this._objects[s].toClipPathSVG(i));
        return this._createBaseClipPathSVGMarkup(r, { reviver: i });
      }
    }), a.Group.fromObject = function (i, r) {
      var s = i.objects, u = a.util.object.clone(i, !0);
      delete u.objects, typeof s != "string" ? a.util.enlivenObjects(s, function (h) {
        a.util.enlivenObjects([i.clipPath], function (p) {
          var m = a.util.object.clone(i, !0);
          m.clipPath = p[0], delete m.objects, r && r(new a.Group(h, m, !0));
        });
      }) : a.loadSVGFromURL(s, function (h) {
        var p = a.util.groupSVGElements(h, i, s);
        p.set(u), r && r(p);
      });
    });
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {});
    a.ActiveSelection || (a.ActiveSelection = a.util.createClass(a.Group, {
      type: "activeSelection", initialize: function (c, f) {
        f = f || {}, this._objects = c || [];
        for (var i = this._objects.length; i--;) this._objects[i].group = this;
        f.originX && (this.originX = f.originX), f.originY && (this.originY = f.originY), this._calcBounds(), this._updateObjectsCoords(), a.Object.prototype.initialize.call(this, f), this.setCoords();
      }, toGroup: function () {
        var c = this._objects.concat();
        this._objects = [];
        var f = a.Object.prototype.toObject.call(this), i = new a.Group([]);
        if (delete f.type, i.set(f), c.forEach(function (s) {
          s.canvas.remove(s), s.group = i;
        }), i._objects = c, !this.canvas) return i;
        var r = this.canvas;
        return r.add(i), r._activeObject = i, i.setCoords(), i;
      }, onDeselect: function () {
        return this.destroy(), !1;
      }, toString: function () {
        return "#<fabric.ActiveSelection: (" + this.complexity() + ")>";
      }, shouldCache: function () {
        return !1;
      }, isOnACache: function () {
        return !1;
      }, _renderControls: function (c, f, i) {
        c.save(), c.globalAlpha = this.isMoving ? this.borderOpacityWhenMoving : 1, this.callSuper("_renderControls", c, f), (i = i || {}).hasControls === void 0 && (i.hasControls = !1), i.forActiveSelection = !0;
        for (var r = 0, s = this._objects.length; r < s; r++) this._objects[r]._renderControls(c, i);
        c.restore();
      }
    }), a.ActiveSelection.fromObject = function (c, f) {
      a.util.enlivenObjects(c.objects, function (i) {
        delete c.objects, f && f(new a.ActiveSelection(i, c, !0));
      });
    });
  }(t), function (l) {
    var a = v.util.object.extend;
    l.fabric || (l.fabric = {}), l.fabric.Image ? v.warn("fabric.Image is already defined.") : (v.Image = v.util.createClass(v.Object, {
      type: "image", strokeWidth: 0, srcFromAttribute: !1, _lastScaleX: 1, _lastScaleY: 1, _filterScalingX: 1, _filterScalingY: 1, minimumScaleTrigger: 0.5, stateProperties: v.Object.prototype.stateProperties.concat("cropX", "cropY"), cacheProperties: v.Object.prototype.cacheProperties.concat("cropX", "cropY"), cacheKey: "", cropX: 0, cropY: 0, imageSmoothing: !0, initialize: function (c, f) {
        f || (f = {}), this.filters = [], this.cacheKey = "texture" + v.Object.__uid++, this.callSuper("initialize", f), this._initElement(c, f);
      }, getElement: function () {
        return this._element || {};
      }, setElement: function (c, f) {
        return this.removeTexture(this.cacheKey), this.removeTexture(this.cacheKey + "_filtered"), this._element = c, this._originalElement = c, this._initConfig(f), this.filters.length !== 0 && this.applyFilters(), this.resizeFilter && this.applyResizeFilters(), this;
      }, removeTexture: function (c) {
        var f = v.filterBackend;
        f && f.evictCachesForKey && f.evictCachesForKey(c);
      }, dispose: function () {
        this.removeTexture(this.cacheKey), this.removeTexture(this.cacheKey + "_filtered"), this._cacheContext = void 0, ["_originalElement", "_element", "_filteredEl", "_cacheCanvas"].forEach((function (c) {
          v.util.cleanUpJsdomNode(this[c]), this[c] = void 0;
        }).bind(this));
      }, getCrossOrigin: function () {
        return this._originalElement && (this._originalElement.crossOrigin || null);
      }, getOriginalSize: function () {
        var c = this.getElement();
        return { width: c.naturalWidth || c.width, height: c.naturalHeight || c.height };
      }, _stroke: function (c) {
        if (this.stroke && this.strokeWidth !== 0) {
          var f = this.width / 2, i = this.height / 2;
          c.beginPath(), c.moveTo(-f, -i), c.lineTo(f, -i), c.lineTo(f, i), c.lineTo(-f, i), c.lineTo(-f, -i), c.closePath();
        }
      }, toObject: function (c) {
        var f = [];
        this.filters.forEach(function (r) {
          r && f.push(r.toObject());
        });
        var i = a(this.callSuper("toObject", ["cropX", "cropY"].concat(c)), { src: this.getSrc(), crossOrigin: this.getCrossOrigin(), filters: f });
        return this.resizeFilter && (i.resizeFilter = this.resizeFilter.toObject()), i;
      }, hasCrop: function () {
        return this.cropX || this.cropY || this.width < this._element.width || this.height < this._element.height;
      }, _toSVG: function () {
        var c, f = [], i = [], r = this._element, s = -this.width / 2, u = -this.height / 2, h = "", p = "";
        if (!r) return [];
        if (this.hasCrop()) {
          var m = v.Object.__uid++;
          f.push('<clipPath id="imageCrop_' + m + `">
`, '    <rect x="' + s + '" y="' + u + '" width="' + this.width + '" height="' + this.height + `" />
`, `</clipPath>
`), h = ' clip-path="url(#imageCrop_' + m + ')" ';
        }
        if (this.imageSmoothing || (p = '" image-rendering="optimizeSpeed'), i.push("    <image ", "COMMON_PARTS", 'xlink:href="', this.getSvgSrc(!0), '" x="', s - this.cropX, '" y="', u - this.cropY, '" width="', r.width || r.naturalWidth, '" height="', r.height || r.height, p, '"', h, `></image>
`), this.stroke || this.strokeDashArray) {
          var y = this.fill;
          this.fill = null, c = ["    <rect ", 'x="', s, '" y="', u, '" width="', this.width, '" height="', this.height, '" style="', this.getSvgStyles(), `"/>
`], this.fill = y;
        }
        return f = this.paintFirst !== "fill" ? f.concat(c, i) : f.concat(i, c);
      }, getSrc: function (c) {
        var f = c ? this._element : this._originalElement;
        return f ? f.toDataURL ? f.toDataURL() : this.srcFromAttribute ? f.getAttribute("src") : f.src : this.src || "";
      }, setSrc: function (c, f, i) {
        return v.util.loadImage(c, function (r, s) {
          this.setElement(r, i), this._setWidthHeight(), f && f(this, s);
        }, this, i && i.crossOrigin), this;
      }, toString: function () {
        return '#<fabric.Image: { src: "' + this.getSrc() + '" }>';
      }, applyResizeFilters: function () {
        var c = this.resizeFilter, f = this.minimumScaleTrigger, i = this.getTotalObjectScaling(), r = i.scaleX, s = i.scaleY, u = this._filteredEl || this._originalElement;
        if (this.group && this.set("dirty", !0), !c || r > f && s > f) return this._element = u, this._filterScalingX = 1, this._filterScalingY = 1, this._lastScaleX = r, void (this._lastScaleY = s);
        v.filterBackend || (v.filterBackend = v.initFilterBackend());
        var h = v.util.createCanvasElement(), p = this._filteredEl ? this.cacheKey + "_filtered" : this.cacheKey, m = u.width, y = u.height;
        h.width = m, h.height = y, this._element = h, this._lastScaleX = c.scaleX = r, this._lastScaleY = c.scaleY = s, v.filterBackend.applyFilters([c], u, m, y, this._element, p), this._filterScalingX = h.width / this._originalElement.width, this._filterScalingY = h.height / this._originalElement.height;
      }, applyFilters: function (c) {
        if (c = (c = c || this.filters || []).filter(function (u) {
          return u && !u.isNeutralState();
        }), this.set("dirty", !0), this.removeTexture(this.cacheKey + "_filtered"), c.length === 0) return this._element = this._originalElement, this._filteredEl = null, this._filterScalingX = 1, this._filterScalingY = 1, this;
        var f = this._originalElement, i = f.naturalWidth || f.width, r = f.naturalHeight || f.height;
        if (this._element === this._originalElement) {
          var s = v.util.createCanvasElement();
          s.width = i, s.height = r, this._element = s, this._filteredEl = s;
        } else this._element = this._filteredEl, this._filteredEl.getContext("2d").clearRect(0, 0, i, r), this._lastScaleX = 1, this._lastScaleY = 1;
        return v.filterBackend || (v.filterBackend = v.initFilterBackend()), v.filterBackend.applyFilters(c, this._originalElement, i, r, this._element, this.cacheKey), this._originalElement.width === this._element.width && this._originalElement.height === this._element.height || (this._filterScalingX = this._element.width / this._originalElement.width, this._filterScalingY = this._element.height / this._originalElement.height), this;
      }, _render: function (c) {
        v.util.setImageSmoothing(c, this.imageSmoothing), this.isMoving !== !0 && this.resizeFilter && this._needsResize() && this.applyResizeFilters(), this._stroke(c), this._renderPaintInOrder(c);
      }, drawCacheOnCanvas: function (c) {
        v.util.setImageSmoothing(c, this.imageSmoothing), v.Object.prototype.drawCacheOnCanvas.call(this, c);
      }, shouldCache: function () {
        return this.needsItsOwnCache();
      }, _renderFill: function (c) {
        var f = this._element;
        if (f) {
          var i = this._filterScalingX, r = this._filterScalingY, s = this.width, u = this.height, h = Math.min, p = Math.max, m = p(this.cropX, 0), y = p(this.cropY, 0), x = f.naturalWidth || f.width, S = f.naturalHeight || f.height, k = m * i, D = y * r, M = h(s * i, x - k), C = h(u * r, S - D), O = -s / 2, N = -u / 2, W = h(s, x / i - m), X = h(u, S / r - y);
          f && c.drawImage(f, k, D, M, C, O, N, W, X);
        }
      }, _needsResize: function () {
        var c = this.getTotalObjectScaling();
        return c.scaleX !== this._lastScaleX || c.scaleY !== this._lastScaleY;
      }, _resetWidthHeight: function () {
        this.set(this.getOriginalSize());
      }, _initElement: function (c, f) {
        this.setElement(v.util.getById(c), f), v.util.addClass(this.getElement(), v.Image.CSS_CANVAS);
      }, _initConfig: function (c) {
        c || (c = {}), this.setOptions(c), this._setWidthHeight(c);
      }, _initFilters: function (c, f) {
        c && c.length ? v.util.enlivenObjects(c, function (i) {
          f && f(i);
        }, "fabric.Image.filters") : f && f();
      }, _setWidthHeight: function (c) {
        c || (c = {});
        var f = this.getElement();
        this.width = c.width || f.naturalWidth || f.width || 0, this.height = c.height || f.naturalHeight || f.height || 0;
      }, parsePreserveAspectRatioAttribute: function () {
        var c, f = v.util.parsePreserveAspectRatioAttribute(this.preserveAspectRatio || ""), i = this._element.width, r = this._element.height, s = 1, u = 1, h = 0, p = 0, m = 0, y = 0, x = this.width, S = this.height, k = { width: x, height: S };
        return !f || f.alignX === "none" && f.alignY === "none" ? (s = x / i, u = S / r) : (f.meetOrSlice === "meet" && (c = (x - i * (s = u = v.util.findScaleToFit(this._element, k))) / 2, f.alignX === "Min" && (h = -c), f.alignX === "Max" && (h = c), c = (S - r * u) / 2, f.alignY === "Min" && (p = -c), f.alignY === "Max" && (p = c)), f.meetOrSlice === "slice" && (c = i - x / (s = u = v.util.findScaleToCover(this._element, k)), f.alignX === "Mid" && (m = c / 2), f.alignX === "Max" && (m = c), c = r - S / u, f.alignY === "Mid" && (y = c / 2), f.alignY === "Max" && (y = c), i = x / s, r = S / u)), { width: i, height: r, scaleX: s, scaleY: u, offsetLeft: h, offsetTop: p, cropX: m, cropY: y };
      }
    }), v.Image.CSS_CANVAS = "canvas-img", v.Image.prototype.getSvgSrc = v.Image.prototype.getSrc, v.Image.fromObject = function (c, f) {
      var i = v.util.object.clone(c);
      v.util.loadImage(i.src, function (r, s) {
        s ? f && f(null, !0) : v.Image.prototype._initFilters.call(i, i.filters, function (u) {
          i.filters = u || [], v.Image.prototype._initFilters.call(i, [i.resizeFilter], function (h) {
            i.resizeFilter = h[0], v.util.enlivenObjects([i.clipPath], function (p) {
              i.clipPath = p[0];
              var m = new v.Image(r, i);
              f(m, !1);
            });
          });
        });
      }, null, i.crossOrigin);
    }, v.Image.fromURL = function (c, f, i) {
      v.util.loadImage(c, function (r, s) {
        f && f(new v.Image(r, i), s);
      }, null, i && i.crossOrigin);
    }, v.Image.ATTRIBUTE_NAMES = v.SHARED_ATTRIBUTES.concat("x y width height preserveAspectRatio xlink:href crossOrigin image-rendering".split(" ")), v.Image.fromElement = function (c, f, i) {
      var r = v.parseAttributes(c, v.Image.ATTRIBUTE_NAMES);
      v.Image.fromURL(r["xlink:href"], f, a(i ? v.util.object.clone(i) : {}, r));
    });
  }(t), v.util.object.extend(v.Object.prototype, {
    _getAngleValueForStraighten: function () {
      var l = this.angle % 360;
      return l > 0 ? 90 * Math.round((l - 1) / 90) : 90 * Math.round(l / 90);
    }, straighten: function () {
      return this.rotate(this._getAngleValueForStraighten()), this;
    }, fxStraighten: function (l) {
      var a = function () {
      }, c = (l = l || {}).onComplete || a, f = l.onChange || a, i = this;
      return v.util.animate({
        startValue: this.get("angle"), endValue: this._getAngleValueForStraighten(), duration: this.FX_DURATION, onChange: function (r) {
          i.rotate(r), f();
        }, onComplete: function () {
          i.setCoords(), c();
        }
      }), this;
    }
  }), v.util.object.extend(v.StaticCanvas.prototype, {
    straightenObject: function (l) {
      return l.straighten(), this.requestRenderAll(), this;
    }, fxStraightenObject: function (l) {
      return l.fxStraighten({ onChange: this.requestRenderAllBound }), this;
    }
  }), function () {
    function l(c, f) {
      var i = "precision " + f + ` float;
void main(){}`, r = c.createShader(c.FRAGMENT_SHADER);
      return c.shaderSource(r, i), c.compileShader(r), !!c.getShaderParameter(r, c.COMPILE_STATUS);
    }
    function a(c) {
      c && c.tileSize && (this.tileSize = c.tileSize), this.setupGLContext(this.tileSize, this.tileSize), this.captureGPUInfo();
    }
    v.isWebglSupported = function (c) {
      if (v.isLikelyNode) return !1;
      c = c || v.WebglFilterBackend.prototype.tileSize;
      var f = document.createElement("canvas"), i = f.getContext("webgl") || f.getContext("experimental-webgl"), r = !1;
      if (i) {
        v.maxTextureSize = i.getParameter(i.MAX_TEXTURE_SIZE), r = v.maxTextureSize >= c;
        for (var s = ["highp", "mediump", "lowp"], u = 0; u < 3; u++) if (l(i, s[u])) {
          v.webGlPrecision = s[u];
          break;
        }
      }
      return this.isSupported = r, r;
    }, v.WebglFilterBackend = a, a.prototype = {
      tileSize: 2048, resources: {}, setupGLContext: function (c, f) {
        this.dispose(), this.createWebGLCanvas(c, f), this.aPosition = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]), this.chooseFastestCopyGLTo2DMethod(c, f);
      }, chooseFastestCopyGLTo2DMethod: function (c, f) {
        var i, r = window.performance !== void 0;
        try {
          new ImageData(1, 1), i = !0;
        } catch {
          i = !1;
        }
        var s = typeof ArrayBuffer < "u", u = typeof Uint8ClampedArray < "u";
        if (r && i && s && u) {
          var h = v.util.createCanvasElement(), p = new ArrayBuffer(c * f * 4);
          if (v.forceGLPutImageData) return this.imageBuffer = p, void (this.copyGLTo2D = It);
          var m, y, x = { imageBuffer: p, destinationWidth: c, destinationHeight: f, targetCanvas: h };
          h.width = c, h.height = f, m = window.performance.now(), Ht.call(x, this.gl, x), y = window.performance.now() - m, m = window.performance.now(), It.call(x, this.gl, x), y > window.performance.now() - m ? (this.imageBuffer = p, this.copyGLTo2D = It) : this.copyGLTo2D = Ht;
        }
      }, createWebGLCanvas: function (c, f) {
        var i = v.util.createCanvasElement();
        i.width = c, i.height = f;
        var r = { alpha: !0, premultipliedAlpha: !1, depth: !1, stencil: !1, antialias: !1 }, s = i.getContext("webgl", r);
        s || (s = i.getContext("experimental-webgl", r)), s && (s.clearColor(0, 0, 0, 0), this.canvas = i, this.gl = s);
      }, applyFilters: function (c, f, i, r, s, u) {
        var h, p = this.gl;
        u && (h = this.getCachedTexture(u, f));
        var m = { originalWidth: f.width || f.originalWidth, originalHeight: f.height || f.originalHeight, sourceWidth: i, sourceHeight: r, destinationWidth: i, destinationHeight: r, context: p, sourceTexture: this.createTexture(p, i, r, !h && f), targetTexture: this.createTexture(p, i, r), originalTexture: h || this.createTexture(p, i, r, !h && f), passes: c.length, webgl: !0, aPosition: this.aPosition, programCache: this.programCache, pass: 0, filterBackend: this, targetCanvas: s }, y = p.createFramebuffer();
        return p.bindFramebuffer(p.FRAMEBUFFER, y), c.forEach(function (x) {
          x && x.applyTo(m);
        }), function (x) {
          var S = x.targetCanvas, k = S.width, D = S.height, M = x.destinationWidth, C = x.destinationHeight;
          k === M && D === C || (S.width = M, S.height = C);
        }(m), this.copyGLTo2D(p, m), p.bindTexture(p.TEXTURE_2D, null), p.deleteTexture(m.sourceTexture), p.deleteTexture(m.targetTexture), p.deleteFramebuffer(y), s.getContext("2d").setTransform(1, 0, 0, 1, 0, 0), m;
      }, dispose: function () {
        this.canvas && (this.canvas = null, this.gl = null), this.clearWebGLCaches();
      }, clearWebGLCaches: function () {
        this.programCache = {}, this.textureCache = {};
      }, createTexture: function (c, f, i, r) {
        var s = c.createTexture();
        return c.bindTexture(c.TEXTURE_2D, s), c.texParameteri(c.TEXTURE_2D, c.TEXTURE_MAG_FILTER, c.NEAREST), c.texParameteri(c.TEXTURE_2D, c.TEXTURE_MIN_FILTER, c.NEAREST), c.texParameteri(c.TEXTURE_2D, c.TEXTURE_WRAP_S, c.CLAMP_TO_EDGE), c.texParameteri(c.TEXTURE_2D, c.TEXTURE_WRAP_T, c.CLAMP_TO_EDGE), r ? c.texImage2D(c.TEXTURE_2D, 0, c.RGBA, c.RGBA, c.UNSIGNED_BYTE, r) : c.texImage2D(c.TEXTURE_2D, 0, c.RGBA, f, i, 0, c.RGBA, c.UNSIGNED_BYTE, null), s;
      }, getCachedTexture: function (c, f) {
        if (this.textureCache[c]) return this.textureCache[c];
        var i = this.createTexture(this.gl, f.width, f.height, f);
        return this.textureCache[c] = i, i;
      }, evictCachesForKey: function (c) {
        this.textureCache[c] && (this.gl.deleteTexture(this.textureCache[c]), delete this.textureCache[c]);
      }, copyGLTo2D: Ht, captureGPUInfo: function () {
        if (this.gpuInfo) return this.gpuInfo;
        var c = this.gl, f = { renderer: "", vendor: "" };
        if (!c) return f;
        var i = c.getExtension("WEBGL_debug_renderer_info");
        if (i) {
          var r = c.getParameter(i.UNMASKED_RENDERER_WEBGL), s = c.getParameter(i.UNMASKED_VENDOR_WEBGL);
          r && (f.renderer = r.toLowerCase()), s && (f.vendor = s.toLowerCase());
        }
        return this.gpuInfo = f, f;
      }
    };
  }(), function () {
    var l = function () {
    };
    function a() {
    }
    v.Canvas2dFilterBackend = a, a.prototype = {
      evictCachesForKey: l, dispose: l, clearWebGLCaches: l, resources: {}, applyFilters: function (c, f, i, r, s) {
        var u = s.getContext("2d");
        u.drawImage(f, 0, 0, i, r);
        var h = { sourceWidth: i, sourceHeight: r, imageData: u.getImageData(0, 0, i, r), originalEl: f, originalImageData: u.getImageData(0, 0, i, r), canvasEl: s, ctx: u, filterBackend: this };
        return c.forEach(function (p) {
          p.applyTo(h);
        }), h.imageData.width === i && h.imageData.height === r || (s.width = h.imageData.width, s.height = h.imageData.height), u.putImageData(h.imageData, 0, 0), h;
      }
    };
  }(), v.Image = v.Image || {}, v.Image.filters = v.Image.filters || {}, v.Image.filters.BaseFilter = v.util.createClass({
    type: "BaseFilter", vertexSource: `attribute vec2 aPosition;
varying vec2 vTexCoord;
void main() {
vTexCoord = aPosition;
gl_Position = vec4(aPosition * 2.0 - 1.0, 0.0, 1.0);
}`, fragmentSource: `precision highp float;
varying vec2 vTexCoord;
uniform sampler2D uTexture;
void main() {
gl_FragColor = texture2D(uTexture, vTexCoord);
}`, initialize: function (l) {
      l && this.setOptions(l);
    }, setOptions: function (l) {
      for (var a in l) this[a] = l[a];
    }, createProgram: function (l, a, c) {
      a = a || this.fragmentSource, c = c || this.vertexSource, v.webGlPrecision !== "highp" && (a = a.replace(/precision highp float/g, "precision " + v.webGlPrecision + " float"));
      var f = l.createShader(l.VERTEX_SHADER);
      if (l.shaderSource(f, c), l.compileShader(f), !l.getShaderParameter(f, l.COMPILE_STATUS)) throw new Error("Vertex shader compile error for " + this.type + ": " + l.getShaderInfoLog(f));
      var i = l.createShader(l.FRAGMENT_SHADER);
      if (l.shaderSource(i, a), l.compileShader(i), !l.getShaderParameter(i, l.COMPILE_STATUS)) throw new Error("Fragment shader compile error for " + this.type + ": " + l.getShaderInfoLog(i));
      var r = l.createProgram();
      if (l.attachShader(r, f), l.attachShader(r, i), l.linkProgram(r), !l.getProgramParameter(r, l.LINK_STATUS)) throw new Error('Shader link error for "${this.type}" ' + l.getProgramInfoLog(r));
      var s = this.getAttributeLocations(l, r), u = this.getUniformLocations(l, r) || {};
      return u.uStepW = l.getUniformLocation(r, "uStepW"), u.uStepH = l.getUniformLocation(r, "uStepH"), { program: r, attributeLocations: s, uniformLocations: u };
    }, getAttributeLocations: function (l, a) {
      return { aPosition: l.getAttribLocation(a, "aPosition") };
    }, getUniformLocations: function () {
      return {};
    }, sendAttributeData: function (l, a, c) {
      var f = a.aPosition, i = l.createBuffer();
      l.bindBuffer(l.ARRAY_BUFFER, i), l.enableVertexAttribArray(f), l.vertexAttribPointer(f, 2, l.FLOAT, !1, 0, 0), l.bufferData(l.ARRAY_BUFFER, c, l.STATIC_DRAW);
    }, _setupFrameBuffer: function (l) {
      var a, c, f = l.context;
      l.passes > 1 ? (a = l.destinationWidth, c = l.destinationHeight, l.sourceWidth === a && l.sourceHeight === c || (f.deleteTexture(l.targetTexture), l.targetTexture = l.filterBackend.createTexture(f, a, c)), f.framebufferTexture2D(f.FRAMEBUFFER, f.COLOR_ATTACHMENT0, f.TEXTURE_2D, l.targetTexture, 0)) : (f.bindFramebuffer(f.FRAMEBUFFER, null), f.finish());
    }, _swapTextures: function (l) {
      l.passes--, l.pass++;
      var a = l.targetTexture;
      l.targetTexture = l.sourceTexture, l.sourceTexture = a;
    }, isNeutralState: function () {
      var l = this.mainParameter, a = v.Image.filters[this.type].prototype;
      if (l) {
        if (Array.isArray(a[l])) {
          for (var c = a[l].length; c--;) if (this[l][c] !== a[l][c]) return !1;
          return !0;
        }
        return a[l] === this[l];
      }
      return !1;
    }, applyTo: function (l) {
      l.webgl ? (this._setupFrameBuffer(l), this.applyToWebGL(l), this._swapTextures(l)) : this.applyTo2d(l);
    }, retrieveShader: function (l) {
      return l.programCache.hasOwnProperty(this.type) || (l.programCache[this.type] = this.createProgram(l.context)), l.programCache[this.type];
    }, applyToWebGL: function (l) {
      var a = l.context, c = this.retrieveShader(l);
      l.pass === 0 && l.originalTexture ? a.bindTexture(a.TEXTURE_2D, l.originalTexture) : a.bindTexture(a.TEXTURE_2D, l.sourceTexture), a.useProgram(c.program), this.sendAttributeData(a, c.attributeLocations, l.aPosition), a.uniform1f(c.uniformLocations.uStepW, 1 / l.sourceWidth), a.uniform1f(c.uniformLocations.uStepH, 1 / l.sourceHeight), this.sendUniformData(a, c.uniformLocations), a.viewport(0, 0, l.destinationWidth, l.destinationHeight), a.drawArrays(a.TRIANGLE_STRIP, 0, 4);
    }, bindAdditionalTexture: function (l, a, c) {
      l.activeTexture(c), l.bindTexture(l.TEXTURE_2D, a), l.activeTexture(l.TEXTURE0);
    }, unbindAdditionalTexture: function (l, a) {
      l.activeTexture(a), l.bindTexture(l.TEXTURE_2D, null), l.activeTexture(l.TEXTURE0);
    }, getMainParameter: function () {
      return this[this.mainParameter];
    }, setMainParameter: function (l) {
      this[this.mainParameter] = l;
    }, sendUniformData: function () {
    }, createHelpLayer: function (l) {
      if (!l.helpLayer) {
        var a = document.createElement("canvas");
        a.width = l.sourceWidth, a.height = l.sourceHeight, l.helpLayer = a;
      }
    }, toObject: function () {
      var l = { type: this.type }, a = this.mainParameter;
      return a && (l[a] = this[a]), l;
    }, toJSON: function () {
      return this.toObject();
    }
  }), v.Image.filters.BaseFilter.fromObject = function (l, a) {
    var c = new v.Image.filters[l.type](l);
    return a && a(c), c;
  }, function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.ColorMatrix = f(c.BaseFilter, {
      type: "ColorMatrix", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
varying vec2 vTexCoord;
uniform mat4 uColorMatrix;
uniform vec4 uConstants;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
color *= uColorMatrix;
color += uConstants;
gl_FragColor = color;
}`, matrix: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], mainParameter: "matrix", colorsOnly: !0, initialize: function (i) {
        this.callSuper("initialize", i), this.matrix = this.matrix.slice(0);
      }, applyTo2d: function (i) {
        var r, s, u, h, p, m = i.imageData.data, y = m.length, x = this.matrix, S = this.colorsOnly;
        for (p = 0; p < y; p += 4) r = m[p], s = m[p + 1], u = m[p + 2], S ? (m[p] = r * x[0] + s * x[1] + u * x[2] + 255 * x[4], m[p + 1] = r * x[5] + s * x[6] + u * x[7] + 255 * x[9], m[p + 2] = r * x[10] + s * x[11] + u * x[12] + 255 * x[14]) : (h = m[p + 3], m[p] = r * x[0] + s * x[1] + u * x[2] + h * x[3] + 255 * x[4], m[p + 1] = r * x[5] + s * x[6] + u * x[7] + h * x[8] + 255 * x[9], m[p + 2] = r * x[10] + s * x[11] + u * x[12] + h * x[13] + 255 * x[14], m[p + 3] = r * x[15] + s * x[16] + u * x[17] + h * x[18] + 255 * x[19]);
      }, getUniformLocations: function (i, r) {
        return { uColorMatrix: i.getUniformLocation(r, "uColorMatrix"), uConstants: i.getUniformLocation(r, "uConstants") };
      }, sendUniformData: function (i, r) {
        var s = this.matrix, u = [s[0], s[1], s[2], s[3], s[5], s[6], s[7], s[8], s[10], s[11], s[12], s[13], s[15], s[16], s[17], s[18]], h = [s[4], s[9], s[14], s[19]];
        i.uniformMatrix4fv(r.uColorMatrix, !1, u), i.uniform4fv(r.uConstants, h);
      }
    }), a.Image.filters.ColorMatrix.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Brightness = f(c.BaseFilter, {
      type: "Brightness", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform float uBrightness;
varying vec2 vTexCoord;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
color.rgb += uBrightness;
gl_FragColor = color;
}`, brightness: 0, mainParameter: "brightness", applyTo2d: function (i) {
        if (this.brightness !== 0) {
          var r, s = i.imageData.data, u = s.length, h = Math.round(255 * this.brightness);
          for (r = 0; r < u; r += 4) s[r] = s[r] + h, s[r + 1] = s[r + 1] + h, s[r + 2] = s[r + 2] + h;
        }
      }, getUniformLocations: function (i, r) {
        return { uBrightness: i.getUniformLocation(r, "uBrightness") };
      }, sendUniformData: function (i, r) {
        i.uniform1f(r.uBrightness, this.brightness);
      }
    }), a.Image.filters.Brightness.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.object.extend, f = a.Image.filters, i = a.util.createClass;
    f.Convolute = i(f.BaseFilter, {
      type: "Convolute", opaque: !1, matrix: [0, 0, 0, 0, 1, 0, 0, 0, 0], fragmentSource: {
        Convolute_3_1: `precision highp float;
uniform sampler2D uTexture;
uniform float uMatrix[9];
uniform float uStepW;
uniform float uStepH;
varying vec2 vTexCoord;
void main() {
vec4 color = vec4(0, 0, 0, 0);
for (float h = 0.0; h < 3.0; h+=1.0) {
for (float w = 0.0; w < 3.0; w+=1.0) {
vec2 matrixPos = vec2(uStepW * (w - 1), uStepH * (h - 1));
color += texture2D(uTexture, vTexCoord + matrixPos) * uMatrix[int(h * 3.0 + w)];
}
}
gl_FragColor = color;
}`, Convolute_3_0: `precision highp float;
uniform sampler2D uTexture;
uniform float uMatrix[9];
uniform float uStepW;
uniform float uStepH;
varying vec2 vTexCoord;
void main() {
vec4 color = vec4(0, 0, 0, 1);
for (float h = 0.0; h < 3.0; h+=1.0) {
for (float w = 0.0; w < 3.0; w+=1.0) {
vec2 matrixPos = vec2(uStepW * (w - 1.0), uStepH * (h - 1.0));
color.rgb += texture2D(uTexture, vTexCoord + matrixPos).rgb * uMatrix[int(h * 3.0 + w)];
}
}
float alpha = texture2D(uTexture, vTexCoord).a;
gl_FragColor = color;
gl_FragColor.a = alpha;
}`, Convolute_5_1: `precision highp float;
uniform sampler2D uTexture;
uniform float uMatrix[25];
uniform float uStepW;
uniform float uStepH;
varying vec2 vTexCoord;
void main() {
vec4 color = vec4(0, 0, 0, 0);
for (float h = 0.0; h < 5.0; h+=1.0) {
for (float w = 0.0; w < 5.0; w+=1.0) {
vec2 matrixPos = vec2(uStepW * (w - 2.0), uStepH * (h - 2.0));
color += texture2D(uTexture, vTexCoord + matrixPos) * uMatrix[int(h * 5.0 + w)];
}
}
gl_FragColor = color;
}`, Convolute_5_0: `precision highp float;
uniform sampler2D uTexture;
uniform float uMatrix[25];
uniform float uStepW;
uniform float uStepH;
varying vec2 vTexCoord;
void main() {
vec4 color = vec4(0, 0, 0, 1);
for (float h = 0.0; h < 5.0; h+=1.0) {
for (float w = 0.0; w < 5.0; w+=1.0) {
vec2 matrixPos = vec2(uStepW * (w - 2.0), uStepH * (h - 2.0));
color.rgb += texture2D(uTexture, vTexCoord + matrixPos).rgb * uMatrix[int(h * 5.0 + w)];
}
}
float alpha = texture2D(uTexture, vTexCoord).a;
gl_FragColor = color;
gl_FragColor.a = alpha;
}`, Convolute_7_1: `precision highp float;
uniform sampler2D uTexture;
uniform float uMatrix[49];
uniform float uStepW;
uniform float uStepH;
varying vec2 vTexCoord;
void main() {
vec4 color = vec4(0, 0, 0, 0);
for (float h = 0.0; h < 7.0; h+=1.0) {
for (float w = 0.0; w < 7.0; w+=1.0) {
vec2 matrixPos = vec2(uStepW * (w - 3.0), uStepH * (h - 3.0));
color += texture2D(uTexture, vTexCoord + matrixPos) * uMatrix[int(h * 7.0 + w)];
}
}
gl_FragColor = color;
}`, Convolute_7_0: `precision highp float;
uniform sampler2D uTexture;
uniform float uMatrix[49];
uniform float uStepW;
uniform float uStepH;
varying vec2 vTexCoord;
void main() {
vec4 color = vec4(0, 0, 0, 1);
for (float h = 0.0; h < 7.0; h+=1.0) {
for (float w = 0.0; w < 7.0; w+=1.0) {
vec2 matrixPos = vec2(uStepW * (w - 3.0), uStepH * (h - 3.0));
color.rgb += texture2D(uTexture, vTexCoord + matrixPos).rgb * uMatrix[int(h * 7.0 + w)];
}
}
float alpha = texture2D(uTexture, vTexCoord).a;
gl_FragColor = color;
gl_FragColor.a = alpha;
}`, Convolute_9_1: `precision highp float;
uniform sampler2D uTexture;
uniform float uMatrix[81];
uniform float uStepW;
uniform float uStepH;
varying vec2 vTexCoord;
void main() {
vec4 color = vec4(0, 0, 0, 0);
for (float h = 0.0; h < 9.0; h+=1.0) {
for (float w = 0.0; w < 9.0; w+=1.0) {
vec2 matrixPos = vec2(uStepW * (w - 4.0), uStepH * (h - 4.0));
color += texture2D(uTexture, vTexCoord + matrixPos) * uMatrix[int(h * 9.0 + w)];
}
}
gl_FragColor = color;
}`, Convolute_9_0: `precision highp float;
uniform sampler2D uTexture;
uniform float uMatrix[81];
uniform float uStepW;
uniform float uStepH;
varying vec2 vTexCoord;
void main() {
vec4 color = vec4(0, 0, 0, 1);
for (float h = 0.0; h < 9.0; h+=1.0) {
for (float w = 0.0; w < 9.0; w+=1.0) {
vec2 matrixPos = vec2(uStepW * (w - 4.0), uStepH * (h - 4.0));
color.rgb += texture2D(uTexture, vTexCoord + matrixPos).rgb * uMatrix[int(h * 9.0 + w)];
}
}
float alpha = texture2D(uTexture, vTexCoord).a;
gl_FragColor = color;
gl_FragColor.a = alpha;
}` }, retrieveShader: function (r) {
        var s = Math.sqrt(this.matrix.length), u = this.type + "_" + s + "_" + (this.opaque ? 1 : 0), h = this.fragmentSource[u];
        return r.programCache.hasOwnProperty(u) || (r.programCache[u] = this.createProgram(r.context, h)), r.programCache[u];
      }, applyTo2d: function (r) {
        var s, u, h, p, m, y, x, S, k, D, M, C, O, N = r.imageData, W = N.data, X = this.matrix, I = Math.round(Math.sqrt(X.length)), T = Math.floor(I / 2), R = N.width, Q = N.height, K = r.ctx.createImageData(R, Q), U = K.data, B = this.opaque ? 1 : 0;
        for (M = 0; M < Q; M++) for (D = 0; D < R; D++) {
          for (m = 4 * (M * R + D), s = 0, u = 0, h = 0, p = 0, O = 0; O < I; O++) for (C = 0; C < I; C++) y = D + C - T, (x = M + O - T) < 0 || x >= Q || y < 0 || y >= R || (S = 4 * (x * R + y), k = X[O * I + C], s += W[S] * k, u += W[S + 1] * k, h += W[S + 2] * k, B || (p += W[S + 3] * k));
          U[m] = s, U[m + 1] = u, U[m + 2] = h, U[m + 3] = B ? W[m + 3] : p;
        }
        r.imageData = K;
      }, getUniformLocations: function (r, s) {
        return { uMatrix: r.getUniformLocation(s, "uMatrix"), uOpaque: r.getUniformLocation(s, "uOpaque"), uHalfSize: r.getUniformLocation(s, "uHalfSize"), uSize: r.getUniformLocation(s, "uSize") };
      }, sendUniformData: function (r, s) {
        r.uniform1fv(s.uMatrix, this.matrix);
      }, toObject: function () {
        return c(this.callSuper("toObject"), { opaque: this.opaque, matrix: this.matrix });
      }
    }), a.Image.filters.Convolute.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Grayscale = f(c.BaseFilter, {
      type: "Grayscale", fragmentSource: {
        average: `precision highp float;
uniform sampler2D uTexture;
varying vec2 vTexCoord;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
float average = (color.r + color.b + color.g) / 3.0;
gl_FragColor = vec4(average, average, average, color.a);
}`, lightness: `precision highp float;
uniform sampler2D uTexture;
uniform int uMode;
varying vec2 vTexCoord;
void main() {
vec4 col = texture2D(uTexture, vTexCoord);
float average = (max(max(col.r, col.g),col.b) + min(min(col.r, col.g),col.b)) / 2.0;
gl_FragColor = vec4(average, average, average, col.a);
}`, luminosity: `precision highp float;
uniform sampler2D uTexture;
uniform int uMode;
varying vec2 vTexCoord;
void main() {
vec4 col = texture2D(uTexture, vTexCoord);
float average = 0.21 * col.r + 0.72 * col.g + 0.07 * col.b;
gl_FragColor = vec4(average, average, average, col.a);
}` }, mode: "average", mainParameter: "mode", applyTo2d: function (i) {
        var r, s, u = i.imageData.data, h = u.length, p = this.mode;
        for (r = 0; r < h; r += 4) p === "average" ? s = (u[r] + u[r + 1] + u[r + 2]) / 3 : p === "lightness" ? s = (Math.min(u[r], u[r + 1], u[r + 2]) + Math.max(u[r], u[r + 1], u[r + 2])) / 2 : p === "luminosity" && (s = 0.21 * u[r] + 0.72 * u[r + 1] + 0.07 * u[r + 2]), u[r] = s, u[r + 1] = s, u[r + 2] = s;
      }, retrieveShader: function (i) {
        var r = this.type + "_" + this.mode;
        if (!i.programCache.hasOwnProperty(r)) {
          var s = this.fragmentSource[this.mode];
          i.programCache[r] = this.createProgram(i.context, s);
        }
        return i.programCache[r];
      }, getUniformLocations: function (i, r) {
        return { uMode: i.getUniformLocation(r, "uMode") };
      }, sendUniformData: function (i, r) {
        i.uniform1i(r.uMode, 1);
      }, isNeutralState: function () {
        return !1;
      }
    }), a.Image.filters.Grayscale.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Invert = f(c.BaseFilter, {
      type: "Invert", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform int uInvert;
varying vec2 vTexCoord;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
if (uInvert == 1) {
gl_FragColor = vec4(1.0 - color.r,1.0 -color.g,1.0 -color.b,color.a);
} else {
gl_FragColor = color;
}
}`, invert: !0, mainParameter: "invert", applyTo2d: function (i) {
        var r, s = i.imageData.data, u = s.length;
        for (r = 0; r < u; r += 4) s[r] = 255 - s[r], s[r + 1] = 255 - s[r + 1], s[r + 2] = 255 - s[r + 2];
      }, isNeutralState: function () {
        return !this.invert;
      }, getUniformLocations: function (i, r) {
        return { uInvert: i.getUniformLocation(r, "uInvert") };
      }, sendUniformData: function (i, r) {
        i.uniform1i(r.uInvert, this.invert);
      }
    }), a.Image.filters.Invert.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.object.extend, f = a.Image.filters, i = a.util.createClass;
    f.Noise = i(f.BaseFilter, {
      type: "Noise", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform float uStepH;
uniform float uNoise;
uniform float uSeed;
varying vec2 vTexCoord;
float rand(vec2 co, float seed, float vScale) {
return fract(sin(dot(co.xy * vScale ,vec2(12.9898 , 78.233))) * 43758.5453 * (seed + 0.01) / 2.0);
}
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
color.rgb += (0.5 - rand(vTexCoord, uSeed, 0.1 / uStepH)) * uNoise;
gl_FragColor = color;
}`, mainParameter: "noise", noise: 0, applyTo2d: function (r) {
        if (this.noise !== 0) {
          var s, u, h = r.imageData.data, p = h.length, m = this.noise;
          for (s = 0, p = h.length; s < p; s += 4) u = (0.5 - Math.random()) * m, h[s] += u, h[s + 1] += u, h[s + 2] += u;
        }
      }, getUniformLocations: function (r, s) {
        return { uNoise: r.getUniformLocation(s, "uNoise"), uSeed: r.getUniformLocation(s, "uSeed") };
      }, sendUniformData: function (r, s) {
        r.uniform1f(s.uNoise, this.noise / 255), r.uniform1f(s.uSeed, Math.random());
      }, toObject: function () {
        return c(this.callSuper("toObject"), { noise: this.noise });
      }
    }), a.Image.filters.Noise.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Pixelate = f(c.BaseFilter, {
      type: "Pixelate", blocksize: 4, mainParameter: "blocksize", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform float uBlocksize;
uniform float uStepW;
uniform float uStepH;
varying vec2 vTexCoord;
void main() {
float blockW = uBlocksize * uStepW;
float blockH = uBlocksize * uStepW;
int posX = int(vTexCoord.x / blockW);
int posY = int(vTexCoord.y / blockH);
float fposX = float(posX);
float fposY = float(posY);
vec2 squareCoords = vec2(fposX * blockW, fposY * blockH);
vec4 color = texture2D(uTexture, squareCoords);
gl_FragColor = color;
}`, applyTo2d: function (i) {
        var r, s, u, h, p, m, y, x, S, k, D, M = i.imageData, C = M.data, O = M.height, N = M.width;
        for (s = 0; s < O; s += this.blocksize) for (u = 0; u < N; u += this.blocksize) for (h = C[r = 4 * s * N + 4 * u], p = C[r + 1], m = C[r + 2], y = C[r + 3], k = Math.min(s + this.blocksize, O), D = Math.min(u + this.blocksize, N), x = s; x < k; x++) for (S = u; S < D; S++) C[r = 4 * x * N + 4 * S] = h, C[r + 1] = p, C[r + 2] = m, C[r + 3] = y;
      }, isNeutralState: function () {
        return this.blocksize === 1;
      }, getUniformLocations: function (i, r) {
        return { uBlocksize: i.getUniformLocation(r, "uBlocksize"), uStepW: i.getUniformLocation(r, "uStepW"), uStepH: i.getUniformLocation(r, "uStepH") };
      }, sendUniformData: function (i, r) {
        i.uniform1f(r.uBlocksize, this.blocksize);
      }
    }), a.Image.filters.Pixelate.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.object.extend, f = a.Image.filters, i = a.util.createClass;
    f.RemoveColor = i(f.BaseFilter, {
      type: "RemoveColor", color: "#FFFFFF", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform vec4 uLow;
uniform vec4 uHigh;
varying vec2 vTexCoord;
void main() {
gl_FragColor = texture2D(uTexture, vTexCoord);
if(all(greaterThan(gl_FragColor.rgb,uLow.rgb)) && all(greaterThan(uHigh.rgb,gl_FragColor.rgb))) {
gl_FragColor.a = 0.0;
}
}`, distance: 0.02, useAlpha: !1, applyTo2d: function (r) {
        var s, u, h, p, m = r.imageData.data, y = 255 * this.distance, x = new a.Color(this.color).getSource(), S = [x[0] - y, x[1] - y, x[2] - y], k = [x[0] + y, x[1] + y, x[2] + y];
        for (s = 0; s < m.length; s += 4) u = m[s], h = m[s + 1], p = m[s + 2], u > S[0] && h > S[1] && p > S[2] && u < k[0] && h < k[1] && p < k[2] && (m[s + 3] = 0);
      }, getUniformLocations: function (r, s) {
        return { uLow: r.getUniformLocation(s, "uLow"), uHigh: r.getUniformLocation(s, "uHigh") };
      }, sendUniformData: function (r, s) {
        var u = new a.Color(this.color).getSource(), h = parseFloat(this.distance), p = [0 + u[0] / 255 - h, 0 + u[1] / 255 - h, 0 + u[2] / 255 - h, 1], m = [u[0] / 255 + h, u[1] / 255 + h, u[2] / 255 + h, 1];
        r.uniform4fv(s.uLow, p), r.uniform4fv(s.uHigh, m);
      }, toObject: function () {
        return c(this.callSuper("toObject"), { color: this.color, distance: this.distance });
      }
    }), a.Image.filters.RemoveColor.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass, i = { Brownie: [0.5997, 0.34553, -0.27082, 0, 0.186, -0.0377, 0.86095, 0.15059, 0, -0.1449, 0.24113, -0.07441, 0.44972, 0, -0.02965, 0, 0, 0, 1, 0], Vintage: [0.62793, 0.32021, -0.03965, 0, 0.03784, 0.02578, 0.64411, 0.03259, 0, 0.02926, 0.0466, -0.08512, 0.52416, 0, 0.02023, 0, 0, 0, 1, 0], Kodachrome: [1.12855, -0.39673, -0.03992, 0, 0.24991, -0.16404, 1.08352, -0.05498, 0, 0.09698, -0.16786, -0.56034, 1.60148, 0, 0.13972, 0, 0, 0, 1, 0], Technicolor: [1.91252, -0.85453, -0.09155, 0, 0.04624, -0.30878, 1.76589, -0.10601, 0, -0.27589, -0.2311, -0.75018, 1.84759, 0, 0.12137, 0, 0, 0, 1, 0], Polaroid: [1.438, -0.062, -0.062, 0, 0, -0.122, 1.378, -0.122, 0, 0, -0.016, -0.016, 1.483, 0, 0, 0, 0, 0, 1, 0], Sepia: [0.393, 0.769, 0.189, 0, 0, 0.349, 0.686, 0.168, 0, 0, 0.272, 0.534, 0.131, 0, 0, 0, 0, 0, 1, 0], BlackWhite: [1.5, 1.5, 1.5, 0, -1, 1.5, 1.5, 1.5, 0, -1, 1.5, 1.5, 1.5, 0, -1, 0, 0, 0, 1, 0] };
    for (var r in i) c[r] = f(c.ColorMatrix, { type: r, matrix: i[r], mainParameter: !1, colorsOnly: !0 }), a.Image.filters[r].fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric, c = a.Image.filters, f = a.util.createClass;
    c.BlendColor = f(c.BaseFilter, {
      type: "BlendColor", color: "#F95C63", mode: "multiply", alpha: 1, fragmentSource: {
        multiply: `gl_FragColor.rgb *= uColor.rgb;
`, screen: `gl_FragColor.rgb = 1.0 - (1.0 - gl_FragColor.rgb) * (1.0 - uColor.rgb);
`, add: `gl_FragColor.rgb += uColor.rgb;
`, diff: `gl_FragColor.rgb = abs(gl_FragColor.rgb - uColor.rgb);
`, subtract: `gl_FragColor.rgb -= uColor.rgb;
`, lighten: `gl_FragColor.rgb = max(gl_FragColor.rgb, uColor.rgb);
`, darken: `gl_FragColor.rgb = min(gl_FragColor.rgb, uColor.rgb);
`, exclusion: `gl_FragColor.rgb += uColor.rgb - 2.0 * (uColor.rgb * gl_FragColor.rgb);
`, overlay: `if (uColor.r < 0.5) {
gl_FragColor.r *= 2.0 * uColor.r;
} else {
gl_FragColor.r = 1.0 - 2.0 * (1.0 - gl_FragColor.r) * (1.0 - uColor.r);
}
if (uColor.g < 0.5) {
gl_FragColor.g *= 2.0 * uColor.g;
} else {
gl_FragColor.g = 1.0 - 2.0 * (1.0 - gl_FragColor.g) * (1.0 - uColor.g);
}
if (uColor.b < 0.5) {
gl_FragColor.b *= 2.0 * uColor.b;
} else {
gl_FragColor.b = 1.0 - 2.0 * (1.0 - gl_FragColor.b) * (1.0 - uColor.b);
}
`, tint: `gl_FragColor.rgb *= (1.0 - uColor.a);
gl_FragColor.rgb += uColor.rgb;
` }, buildSource: function (i) {
        return `precision highp float;
uniform sampler2D uTexture;
uniform vec4 uColor;
varying vec2 vTexCoord;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
gl_FragColor = color;
if (color.a > 0.0) {
` + this.fragmentSource[i] + `}
}`;
      }, retrieveShader: function (i) {
        var r, s = this.type + "_" + this.mode;
        return i.programCache.hasOwnProperty(s) || (r = this.buildSource(this.mode), i.programCache[s] = this.createProgram(i.context, r)), i.programCache[s];
      }, applyTo2d: function (i) {
        var r, s, u, h, p, m, y, x = i.imageData.data, S = x.length, k = 1 - this.alpha;
        r = (y = new a.Color(this.color).getSource())[0] * this.alpha, s = y[1] * this.alpha, u = y[2] * this.alpha;
        for (var D = 0; D < S; D += 4) switch (h = x[D], p = x[D + 1], m = x[D + 2], this.mode) {
          case "multiply":
            x[D] = h * r / 255, x[D + 1] = p * s / 255, x[D + 2] = m * u / 255;
            break;
          case "screen":
            x[D] = 255 - (255 - h) * (255 - r) / 255, x[D + 1] = 255 - (255 - p) * (255 - s) / 255, x[D + 2] = 255 - (255 - m) * (255 - u) / 255;
            break;
          case "add":
            x[D] = h + r, x[D + 1] = p + s, x[D + 2] = m + u;
            break;
          case "diff":
          case "difference":
            x[D] = Math.abs(h - r), x[D + 1] = Math.abs(p - s), x[D + 2] = Math.abs(m - u);
            break;
          case "subtract":
            x[D] = h - r, x[D + 1] = p - s, x[D + 2] = m - u;
            break;
          case "darken":
            x[D] = Math.min(h, r), x[D + 1] = Math.min(p, s), x[D + 2] = Math.min(m, u);
            break;
          case "lighten":
            x[D] = Math.max(h, r), x[D + 1] = Math.max(p, s), x[D + 2] = Math.max(m, u);
            break;
          case "overlay":
            x[D] = r < 128 ? 2 * h * r / 255 : 255 - 2 * (255 - h) * (255 - r) / 255, x[D + 1] = s < 128 ? 2 * p * s / 255 : 255 - 2 * (255 - p) * (255 - s) / 255, x[D + 2] = u < 128 ? 2 * m * u / 255 : 255 - 2 * (255 - m) * (255 - u) / 255;
            break;
          case "exclusion":
            x[D] = r + h - 2 * r * h / 255, x[D + 1] = s + p - 2 * s * p / 255, x[D + 2] = u + m - 2 * u * m / 255;
            break;
          case "tint":
            x[D] = r + h * k, x[D + 1] = s + p * k, x[D + 2] = u + m * k;
        }
      }, getUniformLocations: function (i, r) {
        return { uColor: i.getUniformLocation(r, "uColor") };
      }, sendUniformData: function (i, r) {
        var s = new a.Color(this.color).getSource();
        s[0] = this.alpha * s[0] / 255, s[1] = this.alpha * s[1] / 255, s[2] = this.alpha * s[2] / 255, s[3] = this.alpha, i.uniform4fv(r.uColor, s);
      }, toObject: function () {
        return { type: this.type, color: this.color, mode: this.mode, alpha: this.alpha };
      }
    }), a.Image.filters.BlendColor.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric, c = a.Image.filters, f = a.util.createClass;
    c.BlendImage = f(c.BaseFilter, {
      type: "BlendImage", image: null, mode: "multiply", alpha: 1, vertexSource: `attribute vec2 aPosition;
varying vec2 vTexCoord;
varying vec2 vTexCoord2;
uniform mat3 uTransformMatrix;
void main() {
vTexCoord = aPosition;
vTexCoord2 = (uTransformMatrix * vec3(aPosition, 1.0)).xy;
gl_Position = vec4(aPosition * 2.0 - 1.0, 0.0, 1.0);
}`, fragmentSource: {
        multiply: `precision highp float;
uniform sampler2D uTexture;
uniform sampler2D uImage;
uniform vec4 uColor;
varying vec2 vTexCoord;
varying vec2 vTexCoord2;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
vec4 color2 = texture2D(uImage, vTexCoord2);
color.rgba *= color2.rgba;
gl_FragColor = color;
}`, mask: `precision highp float;
uniform sampler2D uTexture;
uniform sampler2D uImage;
uniform vec4 uColor;
varying vec2 vTexCoord;
varying vec2 vTexCoord2;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
vec4 color2 = texture2D(uImage, vTexCoord2);
color.a = color2.a;
gl_FragColor = color;
}` }, retrieveShader: function (i) {
        var r = this.type + "_" + this.mode, s = this.fragmentSource[this.mode];
        return i.programCache.hasOwnProperty(r) || (i.programCache[r] = this.createProgram(i.context, s)), i.programCache[r];
      }, applyToWebGL: function (i) {
        var r = i.context, s = this.createTexture(i.filterBackend, this.image);
        this.bindAdditionalTexture(r, s, r.TEXTURE1), this.callSuper("applyToWebGL", i), this.unbindAdditionalTexture(r, r.TEXTURE1);
      }, createTexture: function (i, r) {
        return i.getCachedTexture(r.cacheKey, r._element);
      }, calculateMatrix: function () {
        var i = this.image, r = i._element.width, s = i._element.height;
        return [1 / i.scaleX, 0, 0, 0, 1 / i.scaleY, 0, -i.left / r, -i.top / s, 1];
      }, applyTo2d: function (i) {
        var r, s, u, h, p, m, y, x, S, k, D, M = i.imageData, C = i.filterBackend.resources, O = M.data, N = O.length, W = M.width, X = M.height, I = this.image;
        C.blendImage || (C.blendImage = a.util.createCanvasElement()), k = (S = C.blendImage).getContext("2d"), S.width !== W || S.height !== X ? (S.width = W, S.height = X) : k.clearRect(0, 0, W, X), k.setTransform(I.scaleX, 0, 0, I.scaleY, I.left, I.top), k.drawImage(I._element, 0, 0, W, X), D = k.getImageData(0, 0, W, X).data;
        for (var T = 0; T < N; T += 4) switch (p = O[T], m = O[T + 1], y = O[T + 2], x = O[T + 3], r = D[T], s = D[T + 1], u = D[T + 2], h = D[T + 3], this.mode) {
          case "multiply":
            O[T] = p * r / 255, O[T + 1] = m * s / 255, O[T + 2] = y * u / 255, O[T + 3] = x * h / 255;
            break;
          case "mask":
            O[T + 3] = h;
        }
      }, getUniformLocations: function (i, r) {
        return { uTransformMatrix: i.getUniformLocation(r, "uTransformMatrix"), uImage: i.getUniformLocation(r, "uImage") };
      }, sendUniformData: function (i, r) {
        var s = this.calculateMatrix();
        i.uniform1i(r.uImage, 1), i.uniformMatrix3fv(r.uTransformMatrix, !1, s);
      }, toObject: function () {
        return { type: this.type, image: this.image && this.image.toObject(), mode: this.mode, alpha: this.alpha };
      }
    }), a.Image.filters.BlendImage.fromObject = function (i, r) {
      a.Image.fromObject(i.image, function (s) {
        var u = a.util.object.clone(i);
        u.image = s, r(new a.Image.filters.BlendImage(u));
      });
    };
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = Math.pow, f = Math.floor, i = Math.sqrt, r = Math.abs, s = Math.round, u = Math.sin, h = Math.ceil, p = a.Image.filters, m = a.util.createClass;
    p.Resize = m(p.BaseFilter, {
      type: "Resize", resizeType: "hermite", scaleX: 1, scaleY: 1, lanczosLobes: 3, getUniformLocations: function (y, x) {
        return { uDelta: y.getUniformLocation(x, "uDelta"), uTaps: y.getUniformLocation(x, "uTaps") };
      }, sendUniformData: function (y, x) {
        y.uniform2fv(x.uDelta, this.horizontal ? [1 / this.width, 0] : [0, 1 / this.height]), y.uniform1fv(x.uTaps, this.taps);
      }, retrieveShader: function (y) {
        var x = this.getFilterWindow(), S = this.type + "_" + x;
        if (!y.programCache.hasOwnProperty(S)) {
          var k = this.generateShader(x);
          y.programCache[S] = this.createProgram(y.context, k);
        }
        return y.programCache[S];
      }, getFilterWindow: function () {
        var y = this.tempScale;
        return Math.ceil(this.lanczosLobes / y);
      }, getTaps: function () {
        for (var y = this.lanczosCreate(this.lanczosLobes), x = this.tempScale, S = this.getFilterWindow(), k = new Array(S), D = 1; D <= S; D++) k[D - 1] = y(D * x);
        return k;
      }, generateShader: function (y) {
        for (var x = new Array(y), S = this.fragmentSourceTOP, k = 1; k <= y; k++) x[k - 1] = k + ".0 * uDelta";
        return S += "uniform float uTaps[" + y + `];
`, S += `void main() {
`, S += `  vec4 color = texture2D(uTexture, vTexCoord);
`, S += `  float sum = 1.0;
`, x.forEach(function (D, M) {
          S += "  color += texture2D(uTexture, vTexCoord + " + D + ") * uTaps[" + M + `];
`, S += "  color += texture2D(uTexture, vTexCoord - " + D + ") * uTaps[" + M + `];
`, S += "  sum += 2.0 * uTaps[" + M + `];
`;
        }), S += `  gl_FragColor = color / sum;
`, S += "}";
      }, fragmentSourceTOP: `precision highp float;
uniform sampler2D uTexture;
uniform vec2 uDelta;
varying vec2 vTexCoord;
`, applyTo: function (y) {
        y.webgl ? (y.passes++, this.width = y.sourceWidth, this.horizontal = !0, this.dW = Math.round(this.width * this.scaleX), this.dH = y.sourceHeight, this.tempScale = this.dW / this.width, this.taps = this.getTaps(), y.destinationWidth = this.dW, this._setupFrameBuffer(y), this.applyToWebGL(y), this._swapTextures(y), y.sourceWidth = y.destinationWidth, this.height = y.sourceHeight, this.horizontal = !1, this.dH = Math.round(this.height * this.scaleY), this.tempScale = this.dH / this.height, this.taps = this.getTaps(), y.destinationHeight = this.dH, this._setupFrameBuffer(y), this.applyToWebGL(y), this._swapTextures(y), y.sourceHeight = y.destinationHeight) : this.applyTo2d(y);
      }, isNeutralState: function () {
        return this.scaleX === 1 && this.scaleY === 1;
      }, lanczosCreate: function (y) {
        return function (x) {
          if (x >= y || x <= -y) return 0;
          if (x < 11920929e-14 && x > -11920929e-14) return 1;
          var S = (x *= Math.PI) / y;
          return u(x) / x * u(S) / S;
        };
      }, applyTo2d: function (y) {
        var x = y.imageData, S = this.scaleX, k = this.scaleY;
        this.rcpScaleX = 1 / S, this.rcpScaleY = 1 / k;
        var D, M = x.width, C = x.height, O = s(M * S), N = s(C * k);
        this.resizeType === "sliceHack" ? D = this.sliceByTwo(y, M, C, O, N) : this.resizeType === "hermite" ? D = this.hermiteFastResize(y, M, C, O, N) : this.resizeType === "bilinear" ? D = this.bilinearFiltering(y, M, C, O, N) : this.resizeType === "lanczos" && (D = this.lanczosResize(y, M, C, O, N)), y.imageData = D;
      }, sliceByTwo: function (y, x, S, k, D) {
        var M, C, O = y.imageData, N = 0.5, W = !1, X = !1, I = x * N, T = S * N, R = a.filterBackend.resources, Q = 0, K = 0, U = x, B = 0;
        for (R.sliceByTwo || (R.sliceByTwo = document.createElement("canvas")), ((M = R.sliceByTwo).width < 1.5 * x || M.height < S) && (M.width = 1.5 * x, M.height = S), (C = M.getContext("2d")).clearRect(0, 0, 1.5 * x, S), C.putImageData(O, 0, 0), k = f(k), D = f(D); !W || !X;) x = I, S = T, k < f(I * N) ? I = f(I * N) : (I = k, W = !0), D < f(T * N) ? T = f(T * N) : (T = D, X = !0), C.drawImage(M, Q, K, x, S, U, B, I, T), Q = U, K = B, B += T;
        return C.getImageData(Q, K, k, D);
      }, lanczosResize: function (y, x, S, k, D) {
        var M = y.imageData.data, C = y.ctx.createImageData(k, D), O = C.data, N = this.lanczosCreate(this.lanczosLobes), W = this.rcpScaleX, X = this.rcpScaleY, I = 2 / this.rcpScaleX, T = 2 / this.rcpScaleY, R = h(W * this.lanczosLobes / 2), Q = h(X * this.lanczosLobes / 2), K = {}, U = {}, B = {};
        return function Z(z) {
          var V, et, lt, at, ft, ot, ct, st, yt, gt, Tt;
          for (U.x = (z + 0.5) * W, B.x = f(U.x), V = 0; V < D; V++) {
            for (U.y = (V + 0.5) * X, B.y = f(U.y), ft = 0, ot = 0, ct = 0, st = 0, yt = 0, et = B.x - R; et <= B.x + R; et++) if (!(et < 0 || et >= x)) {
              gt = f(1e3 * r(et - U.x)), K[gt] || (K[gt] = {});
              for (var kt = B.y - Q; kt <= B.y + Q; kt++) kt < 0 || kt >= S || (Tt = f(1e3 * r(kt - U.y)), K[gt][Tt] || (K[gt][Tt] = N(i(c(gt * I, 2) + c(Tt * T, 2)) / 1e3)), (lt = K[gt][Tt]) > 0 && (ft += lt, ot += lt * M[at = 4 * (kt * x + et)], ct += lt * M[at + 1], st += lt * M[at + 2], yt += lt * M[at + 3]));
            }
            O[at = 4 * (V * k + z)] = ot / ft, O[at + 1] = ct / ft, O[at + 2] = st / ft, O[at + 3] = yt / ft;
          }
          return ++z < k ? Z(z) : C;
        }(0);
      }, bilinearFiltering: function (y, x, S, k, D) {
        var M, C, O, N, W, X, I, T, R, Q = 0, K = this.rcpScaleX, U = this.rcpScaleY, B = 4 * (x - 1), Z = y.imageData.data, z = y.ctx.createImageData(k, D), V = z.data;
        for (O = 0; O < D; O++) for (N = 0; N < k; N++) for (W = K * N - (M = f(K * N)), X = U * O - (C = f(U * O)), R = 4 * (C * x + M), I = 0; I < 4; I++) T = Z[R + I] * (1 - W) * (1 - X) + Z[R + 4 + I] * W * (1 - X) + Z[R + B + I] * X * (1 - W) + Z[R + B + 4 + I] * W * X, V[Q++] = T;
        return z;
      }, hermiteFastResize: function (y, x, S, k, D) {
        for (var M = this.rcpScaleX, C = this.rcpScaleY, O = h(M / 2), N = h(C / 2), W = y.imageData.data, X = y.ctx.createImageData(k, D), I = X.data, T = 0; T < D; T++) for (var R = 0; R < k; R++) {
          for (var Q = 4 * (R + T * k), K = 0, U = 0, B = 0, Z = 0, z = 0, V = 0, et = 0, lt = (T + 0.5) * C, at = f(T * C); at < (T + 1) * C; at++) for (var ft = r(lt - (at + 0.5)) / N, ot = (R + 0.5) * M, ct = ft * ft, st = f(R * M); st < (R + 1) * M; st++) {
            var yt = r(ot - (st + 0.5)) / O, gt = i(ct + yt * yt);
            gt > 1 && gt < -1 || (K = 2 * gt * gt * gt - 3 * gt * gt + 1) > 0 && (et += K * W[(yt = 4 * (st + at * x)) + 3], B += K, W[yt + 3] < 255 && (K = K * W[yt + 3] / 250), Z += K * W[yt], z += K * W[yt + 1], V += K * W[yt + 2], U += K);
          }
          I[Q] = Z / U, I[Q + 1] = z / U, I[Q + 2] = V / U, I[Q + 3] = et / B;
        }
        return X;
      }, toObject: function () {
        return { type: this.type, scaleX: this.scaleX, scaleY: this.scaleY, resizeType: this.resizeType, lanczosLobes: this.lanczosLobes };
      }
    }), a.Image.filters.Resize.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Contrast = f(c.BaseFilter, {
      type: "Contrast", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform float uContrast;
varying vec2 vTexCoord;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
float contrastF = 1.015 * (uContrast + 1.0) / (1.0 * (1.015 - uContrast));
color.rgb = contrastF * (color.rgb - 0.5) + 0.5;
gl_FragColor = color;
}`, contrast: 0, mainParameter: "contrast", applyTo2d: function (i) {
        if (this.contrast !== 0) {
          var r, s = i.imageData.data, u = s.length, h = Math.floor(255 * this.contrast), p = 259 * (h + 255) / (255 * (259 - h));
          for (r = 0; r < u; r += 4) s[r] = p * (s[r] - 128) + 128, s[r + 1] = p * (s[r + 1] - 128) + 128, s[r + 2] = p * (s[r + 2] - 128) + 128;
        }
      }, getUniformLocations: function (i, r) {
        return { uContrast: i.getUniformLocation(r, "uContrast") };
      }, sendUniformData: function (i, r) {
        i.uniform1f(r.uContrast, this.contrast);
      }
    }), a.Image.filters.Contrast.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Saturation = f(c.BaseFilter, {
      type: "Saturation", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform float uSaturation;
varying vec2 vTexCoord;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
float rgMax = max(color.r, color.g);
float rgbMax = max(rgMax, color.b);
color.r += rgbMax != color.r ? (rgbMax - color.r) * uSaturation : 0.00;
color.g += rgbMax != color.g ? (rgbMax - color.g) * uSaturation : 0.00;
color.b += rgbMax != color.b ? (rgbMax - color.b) * uSaturation : 0.00;
gl_FragColor = color;
}`, saturation: 0, mainParameter: "saturation", applyTo2d: function (i) {
        if (this.saturation !== 0) {
          var r, s, u = i.imageData.data, h = u.length, p = -this.saturation;
          for (r = 0; r < h; r += 4) s = Math.max(u[r], u[r + 1], u[r + 2]), u[r] += s !== u[r] ? (s - u[r]) * p : 0, u[r + 1] += s !== u[r + 1] ? (s - u[r + 1]) * p : 0, u[r + 2] += s !== u[r + 2] ? (s - u[r + 2]) * p : 0;
        }
      }, getUniformLocations: function (i, r) {
        return { uSaturation: i.getUniformLocation(r, "uSaturation") };
      }, sendUniformData: function (i, r) {
        i.uniform1f(r.uSaturation, -this.saturation);
      }
    }), a.Image.filters.Saturation.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Vibrance = f(c.BaseFilter, {
      type: "Vibrance", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform float uVibrance;
varying vec2 vTexCoord;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
float max = max(color.r, max(color.g, color.b));
float avg = (color.r + color.g + color.b) / 3.0;
float amt = (abs(max - avg) * 2.0) * uVibrance;
color.r += max != color.r ? (max - color.r) * amt : 0.00;
color.g += max != color.g ? (max - color.g) * amt : 0.00;
color.b += max != color.b ? (max - color.b) * amt : 0.00;
gl_FragColor = color;
}`, vibrance: 0, mainParameter: "vibrance", applyTo2d: function (i) {
        if (this.vibrance !== 0) {
          var r, s, u, h, p = i.imageData.data, m = p.length, y = -this.vibrance;
          for (r = 0; r < m; r += 4) s = Math.max(p[r], p[r + 1], p[r + 2]), u = (p[r] + p[r + 1] + p[r + 2]) / 3, h = 2 * Math.abs(s - u) / 255 * y, p[r] += s !== p[r] ? (s - p[r]) * h : 0, p[r + 1] += s !== p[r + 1] ? (s - p[r + 1]) * h : 0, p[r + 2] += s !== p[r + 2] ? (s - p[r + 2]) * h : 0;
        }
      }, getUniformLocations: function (i, r) {
        return { uVibrance: i.getUniformLocation(r, "uVibrance") };
      }, sendUniformData: function (i, r) {
        i.uniform1f(r.uVibrance, -this.vibrance);
      }
    }), a.Image.filters.Vibrance.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Blur = f(c.BaseFilter, {
      type: "Blur", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform vec2 uDelta;
varying vec2 vTexCoord;
const float nSamples = 15.0;
vec3 v3offset = vec3(12.9898, 78.233, 151.7182);
float random(vec3 scale) {
return fract(sin(dot(gl_FragCoord.xyz, scale)) * 43758.5453);
}
void main() {
vec4 color = vec4(0.0);
float total = 0.0;
float offset = random(v3offset);
for (float t = -nSamples; t <= nSamples; t++) {
float percent = (t + offset - 0.5) / nSamples;
float weight = 1.0 - abs(percent);
color += texture2D(uTexture, vTexCoord + uDelta * percent) * weight;
total += weight;
}
gl_FragColor = color / total;
}`, blur: 0, mainParameter: "blur", applyTo: function (i) {
        i.webgl ? (this.aspectRatio = i.sourceWidth / i.sourceHeight, i.passes++, this._setupFrameBuffer(i), this.horizontal = !0, this.applyToWebGL(i), this._swapTextures(i), this._setupFrameBuffer(i), this.horizontal = !1, this.applyToWebGL(i), this._swapTextures(i)) : this.applyTo2d(i);
      }, applyTo2d: function (i) {
        i.imageData = this.simpleBlur(i);
      }, simpleBlur: function (i) {
        var r, s, u = i.filterBackend.resources, h = i.imageData.width, p = i.imageData.height;
        u.blurLayer1 || (u.blurLayer1 = a.util.createCanvasElement(), u.blurLayer2 = a.util.createCanvasElement()), r = u.blurLayer1, s = u.blurLayer2, r.width === h && r.height === p || (s.width = r.width = h, s.height = r.height = p);
        var m, y, x, S, k = r.getContext("2d"), D = s.getContext("2d"), M = 15, C = 0.06 * this.blur * 0.5;
        for (k.putImageData(i.imageData, 0, 0), D.clearRect(0, 0, h, p), S = -15; S <= M; S++) x = C * (y = S / M) * h + (m = (Math.random() - 0.5) / 4), D.globalAlpha = 1 - Math.abs(y), D.drawImage(r, x, m), k.drawImage(s, 0, 0), D.globalAlpha = 1, D.clearRect(0, 0, s.width, s.height);
        for (S = -15; S <= M; S++) x = C * (y = S / M) * p + (m = (Math.random() - 0.5) / 4), D.globalAlpha = 1 - Math.abs(y), D.drawImage(r, m, x), k.drawImage(s, 0, 0), D.globalAlpha = 1, D.clearRect(0, 0, s.width, s.height);
        i.ctx.drawImage(r, 0, 0);
        var O = i.ctx.getImageData(0, 0, r.width, r.height);
        return k.globalAlpha = 1, k.clearRect(0, 0, r.width, r.height), O;
      }, getUniformLocations: function (i, r) {
        return { delta: i.getUniformLocation(r, "uDelta") };
      }, sendUniformData: function (i, r) {
        var s = this.chooseRightDelta();
        i.uniform2fv(r.delta, s);
      }, chooseRightDelta: function () {
        var i, r = 1, s = [0, 0];
        return this.horizontal ? this.aspectRatio > 1 && (r = 1 / this.aspectRatio) : this.aspectRatio < 1 && (r = this.aspectRatio), i = r * this.blur * 0.12, this.horizontal ? s[0] = i : s[1] = i, s;
      }
    }), c.Blur.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Gamma = f(c.BaseFilter, {
      type: "Gamma", fragmentSource: `precision highp float;
uniform sampler2D uTexture;
uniform vec3 uGamma;
varying vec2 vTexCoord;
void main() {
vec4 color = texture2D(uTexture, vTexCoord);
vec3 correction = (1.0 / uGamma);
color.r = pow(color.r, correction.r);
color.g = pow(color.g, correction.g);
color.b = pow(color.b, correction.b);
gl_FragColor = color;
gl_FragColor.rgb *= color.a;
}`, gamma: [1, 1, 1], mainParameter: "gamma", initialize: function (i) {
        this.gamma = [1, 1, 1], c.BaseFilter.prototype.initialize.call(this, i);
      }, applyTo2d: function (i) {
        var r, s = i.imageData.data, u = this.gamma, h = s.length, p = 1 / u[0], m = 1 / u[1], y = 1 / u[2];
        for (this.rVals || (this.rVals = new Uint8Array(256), this.gVals = new Uint8Array(256), this.bVals = new Uint8Array(256)), r = 0, h = 256; r < h; r++) this.rVals[r] = 255 * Math.pow(r / 255, p), this.gVals[r] = 255 * Math.pow(r / 255, m), this.bVals[r] = 255 * Math.pow(r / 255, y);
        for (r = 0, h = s.length; r < h; r += 4) s[r] = this.rVals[s[r]], s[r + 1] = this.gVals[s[r + 1]], s[r + 2] = this.bVals[s[r + 2]];
      }, getUniformLocations: function (i, r) {
        return { uGamma: i.getUniformLocation(r, "uGamma") };
      }, sendUniformData: function (i, r) {
        i.uniform3fv(r.uGamma, this.gamma);
      }
    }), a.Image.filters.Gamma.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.Composed = f(c.BaseFilter, {
      type: "Composed", subFilters: [], initialize: function (i) {
        this.callSuper("initialize", i), this.subFilters = this.subFilters.slice(0);
      }, applyTo: function (i) {
        i.passes += this.subFilters.length - 1, this.subFilters.forEach(function (r) {
          r.applyTo(i);
        });
      }, toObject: function () {
        return a.util.object.extend(this.callSuper("toObject"), {
          subFilters: this.subFilters.map(function (i) {
            return i.toObject();
          })
        });
      }, isNeutralState: function () {
        return !this.subFilters.some(function (i) {
          return !i.isNeutralState();
        });
      }
    }), a.Image.filters.Composed.fromObject = function (i, r) {
      var s = (i.subFilters || []).map(function (h) {
        return new a.Image.filters[h.type](h);
      }), u = new a.Image.filters.Composed({ subFilters: s });
      return r && r(u), u;
    };
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.Image.filters, f = a.util.createClass;
    c.HueRotation = f(c.ColorMatrix, {
      type: "HueRotation", rotation: 0, mainParameter: "rotation", calculateMatrix: function () {
        var i = this.rotation * Math.PI, r = a.util.cos(i), s = a.util.sin(i), u = 1 / 3, h = Math.sqrt(u) * s, p = 1 - r;
        this.matrix = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], this.matrix[0] = r + p / 3, this.matrix[1] = u * p - h, this.matrix[2] = u * p + h, this.matrix[5] = u * p + h, this.matrix[6] = r + u * p, this.matrix[7] = u * p - h, this.matrix[10] = u * p - h, this.matrix[11] = u * p + h, this.matrix[12] = r + u * p;
      }, isNeutralState: function (i) {
        return this.calculateMatrix(), c.BaseFilter.prototype.isNeutralState.call(this, i);
      }, applyTo: function (i) {
        this.calculateMatrix(), c.BaseFilter.prototype.applyTo.call(this, i);
      }
    }), a.Image.filters.HueRotation.fromObject = a.Image.filters.BaseFilter.fromObject;
  }(t), function (l) {
    var a = l.fabric || (l.fabric = {}), c = a.util.object.clone;
    if (a.Text) a.warn("fabric.Text is already defined");
    else {
      var f = "fontFamily fontWeight fontSize text underline overline linethrough textAlign fontStyle lineHeight textBackgroundColor charSpacing styles direction path pathStartOffset pathSide".split(" ");
      a.Text = a.util.createClass(a.Object, {
        _dimensionAffectingProps: ["fontSize", "fontWeight", "fontFamily", "fontStyle", "lineHeight", "text", "charSpacing", "textAlign", "styles", "path", "pathStartOffset", "pathSide"], _reNewline: /\r?\n/, _reSpacesAndTabs: /[ \t\r]/g, _reSpaceAndTab: /[ \t\r]/, _reWords: /\S+/g, type: "text", fontSize: 40, fontWeight: "normal", fontFamily: "Times New Roman", underline: !1, overline: !1, linethrough: !1, textAlign: "left", fontStyle: "normal", lineHeight: 1.16, superscript: { size: 0.6, baseline: -0.35 }, subscript: { size: 0.6, baseline: 0.11 }, textBackgroundColor: "", stateProperties: a.Object.prototype.stateProperties.concat(f), cacheProperties: a.Object.prototype.cacheProperties.concat(f), stroke: null, shadow: null, path: null, pathStartOffset: 0, pathSide: "left", _fontSizeFraction: 0.222, offsets: { underline: 0.1, linethrough: -0.315, overline: -0.88 }, _fontSizeMult: 1.13, charSpacing: 0, styles: null, _measuringContext: null, deltaY: 0, direction: "ltr", _styleProperties: ["stroke", "strokeWidth", "fill", "fontFamily", "fontSize", "fontWeight", "fontStyle", "underline", "overline", "linethrough", "deltaY", "textBackgroundColor"], __charBounds: [], CACHE_FONT_SIZE: 400, MIN_TEXT_WIDTH: 2, initialize: function (i, r) {
          this.styles = r && r.styles || {}, this.text = i, this.__skipDimension = !0, this.callSuper("initialize", r), this.path && this.setPathInfo(), this.__skipDimension = !1, this.initDimensions(), this.setCoords(), this.setupState({ propertySet: "_dimensionAffectingProps" });
        }, setPathInfo: function () {
          var i = this.path;
          i && (i.segmentsInfo = a.util.getPathSegmentsInfo(i.path));
        }, getMeasuringContext: function () {
          return a._measuringContext || (a._measuringContext = this.canvas && this.canvas.contextCache || a.util.createCanvasElement().getContext("2d")), a._measuringContext;
        }, _splitText: function () {
          var i = this._splitTextIntoLines(this.text);
          return this.textLines = i.lines, this._textLines = i.graphemeLines, this._unwrappedTextLines = i._unwrappedLines, this._text = i.graphemeText, i;
        }, initDimensions: function () {
          this.__skipDimension || (this._splitText(), this._clearCache(), this.path ? (this.width = this.path.width, this.height = this.path.height) : (this.width = this.calcTextWidth() || this.cursorWidth || this.MIN_TEXT_WIDTH, this.height = this.calcTextHeight()), this.textAlign.indexOf("justify") !== -1 && this.enlargeSpaces(), this.saveState({ propertySet: "_dimensionAffectingProps" }));
        }, enlargeSpaces: function () {
          for (var i, r, s, u, h, p, m, y = 0, x = this._textLines.length; y < x; y++) if ((this.textAlign === "justify" || y !== x - 1 && !this.isEndOfWrapping(y)) && (u = 0, h = this._textLines[y], (r = this.getLineWidth(y)) < this.width && (m = this.textLines[y].match(this._reSpacesAndTabs)))) {
            s = m.length, i = (this.width - r) / s;
            for (var S = 0, k = h.length; S <= k; S++) p = this.__charBounds[y][S], this._reSpaceAndTab.test(h[S]) ? (p.width += i, p.kernedWidth += i, p.left += u, u += i) : p.left += u;
          }
        }, isEndOfWrapping: function (i) {
          return i === this._textLines.length - 1;
        }, missingNewlineOffset: function () {
          return 1;
        }, toString: function () {
          return "#<fabric.Text (" + this.complexity() + '): { "text": "' + this.text + '", "fontFamily": "' + this.fontFamily + '" }>';
        }, _getCacheCanvasDimensions: function () {
          var i = this.callSuper("_getCacheCanvasDimensions"), r = this.fontSize;
          return i.width += r * i.zoomX, i.height += r * i.zoomY, i;
        }, _render: function (i) {
          var r = this.path;
          r && !r.isNotVisible() && r._render(i), this._setTextStyles(i), this._renderTextLinesBackground(i), this._renderTextDecoration(i, "underline"), this._renderText(i), this._renderTextDecoration(i, "overline"), this._renderTextDecoration(i, "linethrough");
        }, _renderText: function (i) {
          this.paintFirst === "stroke" ? (this._renderTextStroke(i), this._renderTextFill(i)) : (this._renderTextFill(i), this._renderTextStroke(i));
        }, _setTextStyles: function (i, r, s) {
          i.textBaseline = "alphabetic", i.font = this._getFontDeclaration(r, s);
        }, calcTextWidth: function () {
          for (var i = this.getLineWidth(0), r = 1, s = this._textLines.length; r < s; r++) {
            var u = this.getLineWidth(r);
            u > i && (i = u);
          }
          return i;
        }, _renderTextLine: function (i, r, s, u, h, p) {
          this._renderChars(i, r, s, u, h, p);
        }, _renderTextLinesBackground: function (i) {
          if (this.textBackgroundColor || this.styleHas("textBackgroundColor")) {
            for (var r, s, u, h, p, m, y, x = i.fillStyle, S = this._getLeftOffset(), k = this._getTopOffset(), D = 0, M = 0, C = this.path, O = 0, N = this._textLines.length; O < N; O++) if (r = this.getHeightOfLine(O), this.textBackgroundColor || this.styleHas("textBackgroundColor", O)) {
              u = this._textLines[O], s = this._getLineLeftOffset(O), M = 0, D = 0, h = this.getValueOfPropertyAt(O, 0, "textBackgroundColor");
              for (var W = 0, X = u.length; W < X; W++) p = this.__charBounds[O][W], m = this.getValueOfPropertyAt(O, W, "textBackgroundColor"), C ? (i.save(), i.translate(p.renderLeft, p.renderTop), i.rotate(p.angle), i.fillStyle = m, m && i.fillRect(-p.width / 2, -r / this.lineHeight * (1 - this._fontSizeFraction), p.width, r / this.lineHeight), i.restore()) : m !== h ? (y = S + s + D, this.direction === "rtl" && (y = this.width - y - M), i.fillStyle = h, h && i.fillRect(y, k, M, r / this.lineHeight), D = p.left, M = p.width, h = m) : M += p.kernedWidth;
              m && !C && (y = S + s + D, this.direction === "rtl" && (y = this.width - y - M), i.fillStyle = m, i.fillRect(y, k, M, r / this.lineHeight)), k += r;
            } else k += r;
            i.fillStyle = x, this._removeShadow(i);
          }
        }, getFontCache: function (i) {
          var r = i.fontFamily.toLowerCase();
          a.charWidthsCache[r] || (a.charWidthsCache[r] = {});
          var s = a.charWidthsCache[r], u = i.fontStyle.toLowerCase() + "_" + (i.fontWeight + "").toLowerCase();
          return s[u] || (s[u] = {}), s[u];
        }, _measureChar: function (i, r, s, u) {
          var h, p, m, y, x = this.getFontCache(r), S = s + i, k = this._getFontDeclaration(r) === this._getFontDeclaration(u), D = r.fontSize / this.CACHE_FONT_SIZE;
          if (s && x[s] !== void 0 && (m = x[s]), x[i] !== void 0 && (y = h = x[i]), k && x[S] !== void 0 && (y = (p = x[S]) - m), h === void 0 || m === void 0 || p === void 0) {
            var M = this.getMeasuringContext();
            this._setTextStyles(M, r, !0);
          }
          return h === void 0 && (y = h = M.measureText(i).width, x[i] = h), m === void 0 && k && s && (m = M.measureText(s).width, x[s] = m), k && p === void 0 && (p = M.measureText(S).width, x[S] = p, y = p - m), { width: h * D, kernedWidth: y * D };
        }, getHeightOfChar: function (i, r) {
          return this.getValueOfPropertyAt(i, r, "fontSize");
        }, measureLine: function (i) {
          var r = this._measureLine(i);
          return this.charSpacing !== 0 && (r.width -= this._getWidthOfCharSpacing()), r.width < 0 && (r.width = 0), r;
        }, _measureLine: function (i) {
          var r, s, u, h, p, m, y = 0, x = this._textLines[i], S = new Array(x.length), k = 0, D = this.path, M = this.pathSide === "right";
          for (this.__charBounds[i] = S, r = 0; r < x.length; r++) s = x[r], h = this._getGraphemeBox(s, i, r, u), S[r] = h, y += h.kernedWidth, u = s;
          if (S[r] = { left: h ? h.left + h.width : 0, width: 0, kernedWidth: 0, height: this.fontSize }, D) {
            switch (m = D.segmentsInfo[D.segmentsInfo.length - 1].length, (p = a.util.getPointOnPath(D.path, 0, D.segmentsInfo)).x += D.pathOffset.x, p.y += D.pathOffset.y, this.textAlign) {
              case "left":
                k = M ? m - y : 0;
                break;
              case "center":
                k = (m - y) / 2;
                break;
              case "right":
                k = M ? 0 : m - y;
            }
            for (k += this.pathStartOffset * (M ? -1 : 1), r = M ? x.length - 1 : 0; M ? r >= 0 : r < x.length; M ? r-- : r++) h = S[r], k > m ? k %= m : k < 0 && (k += m), this._setGraphemeOnPath(k, h, p), k += h.kernedWidth;
          }
          return { width: y, numOfSpaces: 0 };
        }, _setGraphemeOnPath: function (i, r, s) {
          var u = i + r.kernedWidth / 2, h = this.path, p = a.util.getPointOnPath(h.path, u, h.segmentsInfo);
          r.renderLeft = p.x - s.x, r.renderTop = p.y - s.y, r.angle = p.angle + (this.pathSide === "right" ? Math.PI : 0);
        }, _getGraphemeBox: function (i, r, s, u, h) {
          var p, m = this.getCompleteStyleDeclaration(r, s), y = u ? this.getCompleteStyleDeclaration(r, s - 1) : {}, x = this._measureChar(i, m, u, y), S = x.kernedWidth, k = x.width;
          this.charSpacing !== 0 && (k += p = this._getWidthOfCharSpacing(), S += p);
          var D = { width: k, left: 0, height: m.fontSize, kernedWidth: S, deltaY: m.deltaY };
          if (s > 0 && !h) {
            var M = this.__charBounds[r][s - 1];
            D.left = M.left + M.width + x.kernedWidth - x.width;
          }
          return D;
        }, getHeightOfLine: function (i) {
          if (this.__lineHeights[i]) return this.__lineHeights[i];
          for (var r = this._textLines[i], s = this.getHeightOfChar(i, 0), u = 1, h = r.length; u < h; u++) s = Math.max(this.getHeightOfChar(i, u), s);
          return this.__lineHeights[i] = s * this.lineHeight * this._fontSizeMult;
        }, calcTextHeight: function () {
          for (var i, r = 0, s = 0, u = this._textLines.length; s < u; s++) i = this.getHeightOfLine(s), r += s === u - 1 ? i / this.lineHeight : i;
          return r;
        }, _getLeftOffset: function () {
          return this.direction === "ltr" ? -this.width / 2 : this.width / 2;
        }, _getTopOffset: function () {
          return -this.height / 2;
        }, _renderTextCommon: function (i, r) {
          i.save();
          for (var s = 0, u = this._getLeftOffset(), h = this._getTopOffset(), p = 0, m = this._textLines.length; p < m; p++) {
            var y = this.getHeightOfLine(p), x = y / this.lineHeight, S = this._getLineLeftOffset(p);
            this._renderTextLine(r, i, this._textLines[p], u + S, h + s + x, p), s += y;
          }
          i.restore();
        }, _renderTextFill: function (i) {
          (this.fill || this.styleHas("fill")) && this._renderTextCommon(i, "fillText");
        }, _renderTextStroke: function (i) {
          (this.stroke && this.strokeWidth !== 0 || !this.isEmptyStyles()) && (this.shadow && !this.shadow.affectStroke && this._removeShadow(i), i.save(), this._setLineDash(i, this.strokeDashArray), i.beginPath(), this._renderTextCommon(i, "strokeText"), i.closePath(), i.restore());
        }, _renderChars: function (i, r, s, u, h, p) {
          var m, y, x, S, k, D = this.getHeightOfLine(p), M = this.textAlign.indexOf("justify") !== -1, C = "", O = 0, N = this.path, W = !M && this.charSpacing === 0 && this.isEmptyStyles(p) && !N, X = this.direction === "ltr", I = this.direction === "ltr" ? 1 : -1;
          if (r.save(), h -= D * this._fontSizeFraction / this.lineHeight, W) return r.canvas.setAttribute("dir", X ? "ltr" : "rtl"), r.direction = X ? "ltr" : "rtl", r.textAlign = X ? "left" : "right", this._renderChar(i, r, p, 0, s.join(""), u, h, D), void r.restore();
          for (var T = 0, R = s.length - 1; T <= R; T++) S = T === R || this.charSpacing || N, C += s[T], x = this.__charBounds[p][T], O === 0 ? (u += I * (x.kernedWidth - x.width), O += x.width) : O += x.kernedWidth, M && !S && this._reSpaceAndTab.test(s[T]) && (S = !0), S || (m = m || this.getCompleteStyleDeclaration(p, T), y = this.getCompleteStyleDeclaration(p, T + 1), S = this._hasStyleChanged(m, y)), S && (N ? (r.save(), r.translate(x.renderLeft, x.renderTop), r.rotate(x.angle), this._renderChar(i, r, p, T, C, -O / 2, 0, D), r.restore()) : (k = u, r.canvas.setAttribute("dir", X ? "ltr" : "rtl"), r.direction = X ? "ltr" : "rtl", r.textAlign = X ? "left" : "right", this._renderChar(i, r, p, T, C, k, h, D)), C = "", m = y, u += I * O, O = 0);
          r.restore();
        }, _applyPatternGradientTransformText: function (i) {
          var r, s = a.util.createCanvasElement(), u = this.width + this.strokeWidth, h = this.height + this.strokeWidth;
          return s.width = u, s.height = h, (r = s.getContext("2d")).beginPath(), r.moveTo(0, 0), r.lineTo(u, 0), r.lineTo(u, h), r.lineTo(0, h), r.closePath(), r.translate(u / 2, h / 2), r.fillStyle = i.toLive(r), this._applyPatternGradientTransform(r, i), r.fill(), r.createPattern(s, "no-repeat");
        }, handleFiller: function (i, r, s) {
          var u, h;
          return s.toLive ? s.gradientUnits === "percentage" || s.gradientTransform || s.patternTransform ? (u = -this.width / 2, h = -this.height / 2, i.translate(u, h), i[r] = this._applyPatternGradientTransformText(s), { offsetX: u, offsetY: h }) : (i[r] = s.toLive(i, this), this._applyPatternGradientTransform(i, s)) : (i[r] = s, { offsetX: 0, offsetY: 0 });
        }, _setStrokeStyles: function (i, r) {
          return i.lineWidth = r.strokeWidth, i.lineCap = this.strokeLineCap, i.lineDashOffset = this.strokeDashOffset, i.lineJoin = this.strokeLineJoin, i.miterLimit = this.strokeMiterLimit, this.handleFiller(i, "strokeStyle", r.stroke);
        }, _setFillStyles: function (i, r) {
          return this.handleFiller(i, "fillStyle", r.fill);
        }, _renderChar: function (i, r, s, u, h, p, m) {
          var y, x, S = this._getStyleDeclaration(s, u), k = this.getCompleteStyleDeclaration(s, u), D = i === "fillText" && k.fill, M = i === "strokeText" && k.stroke && k.strokeWidth;
          (M || D) && (r.save(), D && (y = this._setFillStyles(r, k)), M && (x = this._setStrokeStyles(r, k)), r.font = this._getFontDeclaration(k), S && S.textBackgroundColor && this._removeShadow(r), S && S.deltaY && (m += S.deltaY), D && r.fillText(h, p - y.offsetX, m - y.offsetY), M && r.strokeText(h, p - x.offsetX, m - x.offsetY), r.restore());
        }, setSuperscript: function (i, r) {
          return this._setScript(i, r, this.superscript);
        }, setSubscript: function (i, r) {
          return this._setScript(i, r, this.subscript);
        }, _setScript: function (i, r, s) {
          var u = this.get2DCursorLocation(i, !0), h = this.getValueOfPropertyAt(u.lineIndex, u.charIndex, "fontSize"), p = this.getValueOfPropertyAt(u.lineIndex, u.charIndex, "deltaY"), m = { fontSize: h * s.size, deltaY: p + h * s.baseline };
          return this.setSelectionStyles(m, i, r), this;
        }, _hasStyleChanged: function (i, r) {
          return i.fill !== r.fill || i.stroke !== r.stroke || i.strokeWidth !== r.strokeWidth || i.fontSize !== r.fontSize || i.fontFamily !== r.fontFamily || i.fontWeight !== r.fontWeight || i.fontStyle !== r.fontStyle || i.deltaY !== r.deltaY;
        }, _hasStyleChangedForSvg: function (i, r) {
          return this._hasStyleChanged(i, r) || i.overline !== r.overline || i.underline !== r.underline || i.linethrough !== r.linethrough;
        }, _getLineLeftOffset: function (i) {
          var r = this.getLineWidth(i), s = this.width - r, u = this.textAlign, h = this.direction, p = 0, m = this.isEndOfWrapping(i);
          return u === "justify" || u === "justify-center" && !m || u === "justify-right" && !m || u === "justify-left" && !m ? 0 : (u === "center" && (p = s / 2), u === "right" && (p = s), u === "justify-center" && (p = s / 2), u === "justify-right" && (p = s), h === "rtl" && (p -= s), p);
        }, _clearCache: function () {
          this.__lineWidths = [], this.__lineHeights = [], this.__charBounds = [];
        }, _shouldClearDimensionCache: function () {
          var i = this._forceClearCache;
          return i || (i = this.hasStateChanged("_dimensionAffectingProps")), i && (this.dirty = !0, this._forceClearCache = !1), i;
        }, getLineWidth: function (i) {
          return this.__lineWidths[i] ? this.__lineWidths[i] : (r = this._textLines[i] === "" ? 0 : this.measureLine(i).width, this.__lineWidths[i] = r, r);
          var r;
        }, _getWidthOfCharSpacing: function () {
          return this.charSpacing !== 0 ? this.fontSize * this.charSpacing / 1e3 : 0;
        }, getValueOfPropertyAt: function (i, r, s) {
          var u = this._getStyleDeclaration(i, r);
          return u && u[s] !== void 0 ? u[s] : this[s];
        }, _renderTextDecoration: function (i, r) {
          if (this[r] || this.styleHas(r)) {
            for (var s, u, h, p, m, y, x, S, k, D, M, C, O, N, W, X, I = this._getLeftOffset(), T = this._getTopOffset(), R = this.path, Q = this._getWidthOfCharSpacing(), K = this.offsets[r], U = 0, B = this._textLines.length; U < B; U++) if (s = this.getHeightOfLine(U), this[r] || this.styleHas(r, U)) {
              x = this._textLines[U], N = s / this.lineHeight, p = this._getLineLeftOffset(U), D = 0, M = 0, S = this.getValueOfPropertyAt(U, 0, r), X = this.getValueOfPropertyAt(U, 0, "fill"), k = T + N * (1 - this._fontSizeFraction), u = this.getHeightOfChar(U, 0), m = this.getValueOfPropertyAt(U, 0, "deltaY");
              for (var Z = 0, z = x.length; Z < z; Z++) if (C = this.__charBounds[U][Z], O = this.getValueOfPropertyAt(U, Z, r), W = this.getValueOfPropertyAt(U, Z, "fill"), h = this.getHeightOfChar(U, Z), y = this.getValueOfPropertyAt(U, Z, "deltaY"), R && O && W) i.save(), i.fillStyle = X, i.translate(C.renderLeft, C.renderTop), i.rotate(C.angle), i.fillRect(-C.kernedWidth / 2, K * h + y, C.kernedWidth, this.fontSize / 15), i.restore();
              else if ((O !== S || W !== X || h !== u || y !== m) && M > 0) {
                var V = I + p + D;
                this.direction === "rtl" && (V = this.width - V - M), S && X && (i.fillStyle = X, i.fillRect(V, k + K * u + m, M, this.fontSize / 15)), D = C.left, M = C.width, S = O, X = W, u = h, m = y;
              } else M += C.kernedWidth;
              V = I + p + D, this.direction === "rtl" && (V = this.width - V - M), i.fillStyle = W, O && W && i.fillRect(V, k + K * u + m, M - Q, this.fontSize / 15), T += s;
            } else T += s;
            this._removeShadow(i);
          }
        }, _getFontDeclaration: function (i, r) {
          var s = i || this, u = this.fontFamily, h = a.Text.genericFonts.indexOf(u.toLowerCase()) > -1, p = u === void 0 || u.indexOf("'") > -1 || u.indexOf(",") > -1 || u.indexOf('"') > -1 || h ? s.fontFamily : '"' + s.fontFamily + '"';
          return [a.isLikelyNode ? s.fontWeight : s.fontStyle, a.isLikelyNode ? s.fontStyle : s.fontWeight, r ? this.CACHE_FONT_SIZE + "px" : s.fontSize + "px", p].join(" ");
        }, render: function (i) {
          this.visible && (this.canvas && this.canvas.skipOffscreen && !this.group && !this.isOnScreen() || (this._shouldClearDimensionCache() && this.initDimensions(), this.callSuper("render", i)));
        }, _splitTextIntoLines: function (i) {
          for (var r = i.split(this._reNewline), s = new Array(r.length), u = [`
`], h = [], p = 0; p < r.length; p++) s[p] = a.util.string.graphemeSplit(r[p]), h = h.concat(s[p], u);
          return h.pop(), { _unwrappedLines: s, lines: r, graphemeText: h, graphemeLines: s };
        }, toObject: function (i) {
          var r = f.concat(i), s = this.callSuper("toObject", r);
          return s.styles = c(this.styles, !0), s.path && (s.path = this.path.toObject()), s;
        }, set: function (i, r) {
          this.callSuper("set", i, r);
          var s = !1, u = !1;
          if (typeof i == "object") for (var h in i) h === "path" && this.setPathInfo(), s = s || this._dimensionAffectingProps.indexOf(h) !== -1, u = u || h === "path";
          else s = this._dimensionAffectingProps.indexOf(i) !== -1, u = i === "path";
          return u && this.setPathInfo(), s && (this.initDimensions(), this.setCoords()), this;
        }, complexity: function () {
          return 1;
        }
      }), a.Text.ATTRIBUTE_NAMES = a.SHARED_ATTRIBUTES.concat("x y dx dy font-family font-style font-weight font-size letter-spacing text-decoration text-anchor".split(" ")), a.Text.DEFAULT_SVG_FONT_SIZE = 16, a.Text.fromElement = function (i, r, s) {
        if (!i) return r(null);
        var u = a.parseAttributes(i, a.Text.ATTRIBUTE_NAMES), h = u.textAnchor || "left";
        if ((s = a.util.object.extend(s ? c(s) : {}, u)).top = s.top || 0, s.left = s.left || 0, u.textDecoration) {
          var p = u.textDecoration;
          p.indexOf("underline") !== -1 && (s.underline = !0), p.indexOf("overline") !== -1 && (s.overline = !0), p.indexOf("line-through") !== -1 && (s.linethrough = !0), delete s.textDecoration;
        }
        "dx" in u && (s.left += u.dx), "dy" in u && (s.top += u.dy), "fontSize" in s || (s.fontSize = a.Text.DEFAULT_SVG_FONT_SIZE);
        var m = "";
        "textContent" in i ? m = i.textContent : "firstChild" in i && i.firstChild !== null && "data" in i.firstChild && i.firstChild.data !== null && (m = i.firstChild.data), m = m.replace(/^\s+|\s+$|\n+/g, "").replace(/\s+/g, " ");
        var y = s.strokeWidth;
        s.strokeWidth = 0;
        var x = new a.Text(m, s), S = x.getScaledHeight() / x.height, k = ((x.height + x.strokeWidth) * x.lineHeight - x.height) * S, D = x.getScaledHeight() + k, M = 0;
        h === "center" && (M = x.getScaledWidth() / 2), h === "right" && (M = x.getScaledWidth()), x.set({ left: x.left - M, top: x.top - (D - x.fontSize * (0.07 + x._fontSizeFraction)) / x.lineHeight, strokeWidth: y !== void 0 ? y : 1 }), r(x);
      }, a.Text.fromObject = function (i, r) {
        var s = c(i), u = i.path;
        return delete s.path, a.Object._fromObject("Text", s, function (h) {
          u ? a.Object._fromObject("Path", u, function (p) {
            h.set("path", p), r(h);
          }, "path") : r(h);
        }, "text");
      }, a.Text.genericFonts = ["sans-serif", "serif", "cursive", "fantasy", "monospace"], a.util.createAccessors && a.util.createAccessors(a.Text);
    }
  }(t), v.util.object.extend(v.Text.prototype, {
    isEmptyStyles: function (l) {
      if (!this.styles || l !== void 0 && !this.styles[l]) return !0;
      var a = l === void 0 ? this.styles : { line: this.styles[l] };
      for (var c in a) for (var f in a[c]) for (var i in a[c][f]) return !1;
      return !0;
    }, styleHas: function (l, a) {
      if (!this.styles || !l || l === "" || a !== void 0 && !this.styles[a]) return !1;
      var c = a === void 0 ? this.styles : { 0: this.styles[a] };
      for (var f in c) for (var i in c[f]) if (c[f][i][l] !== void 0) return !0;
      return !1;
    }, cleanStyle: function (l) {
      if (!this.styles || !l || l === "") return !1;
      var a, c, f = this.styles, i = 0, r = !0, s = 0;
      for (var u in f) {
        for (var h in a = 0, f[u]) {
          var p;
          i++, (p = f[u][h]).hasOwnProperty(l) ? (c ? p[l] !== c && (r = !1) : c = p[l], p[l] === this[l] && delete p[l]) : r = !1, Object.keys(p).length !== 0 ? a++ : delete f[u][h];
        }
        a === 0 && delete f[u];
      }
      for (var m = 0; m < this._textLines.length; m++) s += this._textLines[m].length;
      r && i === s && (this[l] = c, this.removeStyle(l));
    }, removeStyle: function (l) {
      if (this.styles && l && l !== "") {
        var a, c, f, i = this.styles;
        for (c in i) {
          for (f in a = i[c]) delete a[f][l], Object.keys(a[f]).length === 0 && delete a[f];
          Object.keys(a).length === 0 && delete i[c];
        }
      }
    }, _extendStyles: function (l, a) {
      var c = this.get2DCursorLocation(l);
      this._getLineStyle(c.lineIndex) || this._setLineStyle(c.lineIndex), this._getStyleDeclaration(c.lineIndex, c.charIndex) || this._setStyleDeclaration(c.lineIndex, c.charIndex, {}), v.util.object.extend(this._getStyleDeclaration(c.lineIndex, c.charIndex), a);
    }, get2DCursorLocation: function (l, a) {
      l === void 0 && (l = this.selectionStart);
      for (var c = a ? this._unwrappedTextLines : this._textLines, f = c.length, i = 0; i < f; i++) {
        if (l <= c[i].length) return { lineIndex: i, charIndex: l };
        l -= c[i].length + this.missingNewlineOffset(i);
      }
      return { lineIndex: i - 1, charIndex: c[i - 1].length < l ? c[i - 1].length : l };
    }, getSelectionStyles: function (l, a, c) {
      l === void 0 && (l = this.selectionStart || 0), a === void 0 && (a = this.selectionEnd || l);
      for (var f = [], i = l; i < a; i++) f.push(this.getStyleAtPosition(i, c));
      return f;
    }, getStyleAtPosition: function (l, a) {
      var c = this.get2DCursorLocation(l);
      return (a ? this.getCompleteStyleDeclaration(c.lineIndex, c.charIndex) : this._getStyleDeclaration(c.lineIndex, c.charIndex)) || {};
    }, setSelectionStyles: function (l, a, c) {
      a === void 0 && (a = this.selectionStart || 0), c === void 0 && (c = this.selectionEnd || a);
      for (var f = a; f < c; f++) this._extendStyles(f, l);
      return this._forceClearCache = !0, this;
    }, _getStyleDeclaration: function (l, a) {
      var c = this.styles && this.styles[l];
      return c ? c[a] : null;
    }, getCompleteStyleDeclaration: function (l, a) {
      for (var c, f = this._getStyleDeclaration(l, a) || {}, i = {}, r = 0; r < this._styleProperties.length; r++) i[c = this._styleProperties[r]] = f[c] === void 0 ? this[c] : f[c];
      return i;
    }, _setStyleDeclaration: function (l, a, c) {
      this.styles[l][a] = c;
    }, _deleteStyleDeclaration: function (l, a) {
      delete this.styles[l][a];
    }, _getLineStyle: function (l) {
      return !!this.styles[l];
    }, _setLineStyle: function (l) {
      this.styles[l] = {};
    }, _deleteLineStyle: function (l) {
      delete this.styles[l];
    }
  }), function () {
    function l(a) {
      a.textDecoration && (a.textDecoration.indexOf("underline") > -1 && (a.underline = !0), a.textDecoration.indexOf("line-through") > -1 && (a.linethrough = !0), a.textDecoration.indexOf("overline") > -1 && (a.overline = !0), delete a.textDecoration);
    }
    v.IText = v.util.createClass(v.Text, v.Observable, {
      type: "i-text", selectionStart: 0, selectionEnd: 0, selectionColor: "rgba(17,119,255,0.3)", isEditing: !1, editable: !0, editingBorderColor: "rgba(102,153,255,0.25)", cursorWidth: 2, cursorColor: "", cursorDelay: 1e3, cursorDuration: 600, caching: !0, hiddenTextareaContainer: null, _reSpace: /\s|\n/, _currentCursorOpacity: 0, _selectionDirection: null, _abortCursorAnimation: !1, __widthOfSpace: [], inCompositionMode: !1, initialize: function (a, c) {
        this.callSuper("initialize", a, c), this.initBehavior();
      }, setSelectionStart: function (a) {
        a = Math.max(a, 0), this._updateAndFire("selectionStart", a);
      }, setSelectionEnd: function (a) {
        a = Math.min(a, this.text.length), this._updateAndFire("selectionEnd", a);
      }, _updateAndFire: function (a, c) {
        this[a] !== c && (this._fireSelectionChanged(), this[a] = c), this._updateTextarea();
      }, _fireSelectionChanged: function () {
        this.fire("selection:changed"), this.canvas && this.canvas.fire("text:selection:changed", { target: this });
      }, initDimensions: function () {
        this.isEditing && this.initDelayedCursor(), this.clearContextTop(), this.callSuper("initDimensions");
      }, render: function (a) {
        this.clearContextTop(), this.callSuper("render", a), this.cursorOffsetCache = {}, this.renderCursorOrSelection();
      }, _render: function (a) {
        this.callSuper("_render", a);
      }, clearContextTop: function (a) {
        if (this.isEditing && this.canvas && this.canvas.contextTop) {
          var c = this.canvas.contextTop, f = this.canvas.viewportTransform;
          c.save(), c.transform(f[0], f[1], f[2], f[3], f[4], f[5]), this.transform(c), this._clearTextArea(c), a || c.restore();
        }
      }, renderCursorOrSelection: function () {
        if (this.isEditing && this.canvas && this.canvas.contextTop) {
          var a = this._getCursorBoundaries(), c = this.canvas.contextTop;
          this.clearContextTop(!0), this.selectionStart === this.selectionEnd ? this.renderCursor(a, c) : this.renderSelection(a, c), c.restore();
        }
      }, _clearTextArea: function (a) {
        var c = this.width + 4, f = this.height + 4;
        a.clearRect(-c / 2, -f / 2, c, f);
      }, _getCursorBoundaries: function (a) {
        a === void 0 && (a = this.selectionStart);
        var c = this._getLeftOffset(), f = this._getTopOffset(), i = this._getCursorBoundariesOffsets(a);
        return { left: c, top: f, leftOffset: i.left, topOffset: i.top };
      }, _getCursorBoundariesOffsets: function (a) {
        if (this.cursorOffsetCache && "top" in this.cursorOffsetCache) return this.cursorOffsetCache;
        var c, f, i, r, s = 0, u = 0, h = this.get2DCursorLocation(a);
        i = h.charIndex, f = h.lineIndex;
        for (var p = 0; p < f; p++) s += this.getHeightOfLine(p);
        c = this._getLineLeftOffset(f);
        var m = this.__charBounds[f][i];
        return m && (u = m.left), this.charSpacing !== 0 && i === this._textLines[f].length && (u -= this._getWidthOfCharSpacing()), r = { top: s, left: c + (u > 0 ? u : 0) }, this.direction === "rtl" && (r.left *= -1), this.cursorOffsetCache = r, this.cursorOffsetCache;
      }, renderCursor: function (a, c) {
        var f = this.get2DCursorLocation(), i = f.lineIndex, r = f.charIndex > 0 ? f.charIndex - 1 : 0, s = this.getValueOfPropertyAt(i, r, "fontSize"), u = this.scaleX * this.canvas.getZoom(), h = this.cursorWidth / u, p = a.topOffset, m = this.getValueOfPropertyAt(i, r, "deltaY");
        p += (1 - this._fontSizeFraction) * this.getHeightOfLine(i) / this.lineHeight - s * (1 - this._fontSizeFraction), this.inCompositionMode && this.renderSelection(a, c), c.fillStyle = this.cursorColor || this.getValueOfPropertyAt(i, r, "fill"), c.globalAlpha = this.__isMousedown ? 1 : this._currentCursorOpacity, c.fillRect(a.left + a.leftOffset - h / 2, p + a.top + m, h, s);
      }, renderSelection: function (a, c) {
        for (var f = this.inCompositionMode ? this.hiddenTextarea.selectionStart : this.selectionStart, i = this.inCompositionMode ? this.hiddenTextarea.selectionEnd : this.selectionEnd, r = this.textAlign.indexOf("justify") !== -1, s = this.get2DCursorLocation(f), u = this.get2DCursorLocation(i), h = s.lineIndex, p = u.lineIndex, m = s.charIndex < 0 ? 0 : s.charIndex, y = u.charIndex < 0 ? 0 : u.charIndex, x = h; x <= p; x++) {
          var S, k = this._getLineLeftOffset(x) || 0, D = this.getHeightOfLine(x), M = 0, C = 0;
          if (x === h && (M = this.__charBounds[h][m].left), x >= h && x < p) C = r && !this.isEndOfWrapping(x) ? this.width : this.getLineWidth(x) || 5;
          else if (x === p) if (y === 0) C = this.__charBounds[p][y].left;
          else {
            var O = this._getWidthOfCharSpacing();
            C = this.__charBounds[p][y - 1].left + this.__charBounds[p][y - 1].width - O;
          }
          S = D, (this.lineHeight < 1 || x === p && this.lineHeight > 1) && (D /= this.lineHeight);
          var N = a.left + k + M, W = C - M, X = D, I = 0;
          this.inCompositionMode ? (c.fillStyle = this.compositionColor || "black", X = 1, I = D) : c.fillStyle = this.selectionColor, this.direction === "rtl" && (N = this.width - N - W), c.fillRect(N, a.top + a.topOffset + I, W, X), a.topOffset += S;
        }
      }, getCurrentCharFontSize: function () {
        var a = this._getCurrentCharIndex();
        return this.getValueOfPropertyAt(a.l, a.c, "fontSize");
      }, getCurrentCharColor: function () {
        var a = this._getCurrentCharIndex();
        return this.getValueOfPropertyAt(a.l, a.c, "fill");
      }, _getCurrentCharIndex: function () {
        var a = this.get2DCursorLocation(this.selectionStart, !0), c = a.charIndex > 0 ? a.charIndex - 1 : 0;
        return { l: a.lineIndex, c };
      }
    }), v.IText.fromObject = function (a, c) {
      if (l(a), a.styles) for (var f in a.styles) for (var i in a.styles[f]) l(a.styles[f][i]);
      v.Object._fromObject("IText", a, c, "text");
    };
  }(), ht = v.util.object.clone, v.util.object.extend(v.IText.prototype, {
    initBehavior: function () {
      this.initAddedHandler(), this.initRemovedHandler(), this.initCursorSelectionHandlers(), this.initDoubleClickSimulation(), this.mouseMoveHandler = this.mouseMoveHandler.bind(this);
    }, onDeselect: function () {
      this.isEditing && this.exitEditing(), this.selected = !1;
    }, initAddedHandler: function () {
      var l = this;
      this.on("added", function () {
        var a = l.canvas;
        a && (a._hasITextHandlers || (a._hasITextHandlers = !0, l._initCanvasHandlers(a)), a._iTextInstances = a._iTextInstances || [], a._iTextInstances.push(l));
      });
    }, initRemovedHandler: function () {
      var l = this;
      this.on("removed", function () {
        var a = l.canvas;
        a && (a._iTextInstances = a._iTextInstances || [], v.util.removeFromArray(a._iTextInstances, l), a._iTextInstances.length === 0 && (a._hasITextHandlers = !1, l._removeCanvasHandlers(a)));
      });
    }, _initCanvasHandlers: function (l) {
      l._mouseUpITextHandler = function () {
        l._iTextInstances && l._iTextInstances.forEach(function (a) {
          a.__isMousedown = !1;
        });
      }, l.on("mouse:up", l._mouseUpITextHandler);
    }, _removeCanvasHandlers: function (l) {
      l.off("mouse:up", l._mouseUpITextHandler);
    }, _tick: function () {
      this._currentTickState = this._animateCursor(this, 1, this.cursorDuration, "_onTickComplete");
    }, _animateCursor: function (l, a, c, f) {
      var i;
      return i = {
        isAborted: !1, abort: function () {
          this.isAborted = !0;
        }
      }, l.animate("_currentCursorOpacity", a, {
        duration: c, onComplete: function () {
          i.isAborted || l[f]();
        }, onChange: function () {
          l.canvas && l.selectionStart === l.selectionEnd && l.renderCursorOrSelection();
        }, abort: function () {
          return i.isAborted;
        }
      }), i;
    }, _onTickComplete: function () {
      var l = this;
      this._cursorTimeout1 && clearTimeout(this._cursorTimeout1), this._cursorTimeout1 = setTimeout(function () {
        l._currentTickCompleteState = l._animateCursor(l, 0, this.cursorDuration / 2, "_tick");
      }, 100);
    }, initDelayedCursor: function (l) {
      var a = this, c = l ? 0 : this.cursorDelay;
      this.abortCursorAnimation(), this._currentCursorOpacity = 1, this._cursorTimeout2 = setTimeout(function () {
        a._tick();
      }, c);
    }, abortCursorAnimation: function () {
      var l = this._currentTickState || this._currentTickCompleteState, a = this.canvas;
      this._currentTickState && this._currentTickState.abort(), this._currentTickCompleteState && this._currentTickCompleteState.abort(), clearTimeout(this._cursorTimeout1), clearTimeout(this._cursorTimeout2), this._currentCursorOpacity = 0, l && a && a.clearContext(a.contextTop || a.contextContainer);
    }, selectAll: function () {
      return this.selectionStart = 0, this.selectionEnd = this._text.length, this._fireSelectionChanged(), this._updateTextarea(), this;
    }, getSelectedText: function () {
      return this._text.slice(this.selectionStart, this.selectionEnd).join("");
    }, findWordBoundaryLeft: function (l) {
      var a = 0, c = l - 1;
      if (this._reSpace.test(this._text[c])) for (; this._reSpace.test(this._text[c]);) a++, c--;
      for (; /\S/.test(this._text[c]) && c > -1;) a++, c--;
      return l - a;
    }, findWordBoundaryRight: function (l) {
      var a = 0, c = l;
      if (this._reSpace.test(this._text[c])) for (; this._reSpace.test(this._text[c]);) a++, c++;
      for (; /\S/.test(this._text[c]) && c < this._text.length;) a++, c++;
      return l + a;
    }, findLineBoundaryLeft: function (l) {
      for (var a = 0, c = l - 1; !/\n/.test(this._text[c]) && c > -1;) a++, c--;
      return l - a;
    }, findLineBoundaryRight: function (l) {
      for (var a = 0, c = l; !/\n/.test(this._text[c]) && c < this._text.length;) a++, c++;
      return l + a;
    }, searchWordBoundary: function (l, a) {
      for (var c = this._text, f = this._reSpace.test(c[l]) ? l - 1 : l, i = c[f], r = v.reNonWord; !r.test(i) && f > 0 && f < c.length;) i = c[f += a];
      return r.test(i) && (f += a === 1 ? 0 : 1), f;
    }, selectWord: function (l) {
      l = l || this.selectionStart;
      var a = this.searchWordBoundary(l, -1), c = this.searchWordBoundary(l, 1);
      this.selectionStart = a, this.selectionEnd = c, this._fireSelectionChanged(), this._updateTextarea(), this.renderCursorOrSelection();
    }, selectLine: function (l) {
      l = l || this.selectionStart;
      var a = this.findLineBoundaryLeft(l), c = this.findLineBoundaryRight(l);
      return this.selectionStart = a, this.selectionEnd = c, this._fireSelectionChanged(), this._updateTextarea(), this;
    }, enterEditing: function (l) {
      if (!this.isEditing && this.editable) return this.canvas && (this.canvas.calcOffset(), this.exitEditingOnOthers(this.canvas)), this.isEditing = !0, this.initHiddenTextarea(l), this.hiddenTextarea.focus(), this.hiddenTextarea.value = this.text, this._updateTextarea(), this._saveEditingProps(), this._setEditingProps(), this._textBeforeEdit = this.text, this._tick(), this.fire("editing:entered"), this._fireSelectionChanged(), this.canvas ? (this.canvas.fire("text:editing:entered", { target: this }), this.initMouseMoveHandler(), this.canvas.requestRenderAll(), this) : this;
    }, exitEditingOnOthers: function (l) {
      l._iTextInstances && l._iTextInstances.forEach(function (a) {
        a.selected = !1, a.isEditing && a.exitEditing();
      });
    }, initMouseMoveHandler: function () {
      this.canvas.on("mouse:move", this.mouseMoveHandler);
    }, mouseMoveHandler: function (l) {
      if (this.__isMousedown && this.isEditing) {
        var a = this.getSelectionStartFromPointer(l.e), c = this.selectionStart, f = this.selectionEnd;
        (a === this.__selectionStartOnMouseDown && c !== f || c !== a && f !== a) && (a > this.__selectionStartOnMouseDown ? (this.selectionStart = this.__selectionStartOnMouseDown, this.selectionEnd = a) : (this.selectionStart = a, this.selectionEnd = this.__selectionStartOnMouseDown), this.selectionStart === c && this.selectionEnd === f || (this.restartCursorIfNeeded(), this._fireSelectionChanged(), this._updateTextarea(), this.renderCursorOrSelection()));
      }
    }, _setEditingProps: function () {
      this.hoverCursor = "text", this.canvas && (this.canvas.defaultCursor = this.canvas.moveCursor = "text"), this.borderColor = this.editingBorderColor, this.hasControls = this.selectable = !1, this.lockMovementX = this.lockMovementY = !0;
    }, fromStringToGraphemeSelection: function (l, a, c) {
      var f = c.slice(0, l), i = v.util.string.graphemeSplit(f).length;
      if (l === a) return { selectionStart: i, selectionEnd: i };
      var r = c.slice(l, a);
      return { selectionStart: i, selectionEnd: i + v.util.string.graphemeSplit(r).length };
    }, fromGraphemeToStringSelection: function (l, a, c) {
      var f = c.slice(0, l).join("").length;
      return l === a ? { selectionStart: f, selectionEnd: f } : { selectionStart: f, selectionEnd: f + c.slice(l, a).join("").length };
    }, _updateTextarea: function () {
      if (this.cursorOffsetCache = {}, this.hiddenTextarea) {
        if (!this.inCompositionMode) {
          var l = this.fromGraphemeToStringSelection(this.selectionStart, this.selectionEnd, this._text);
          this.hiddenTextarea.selectionStart = l.selectionStart, this.hiddenTextarea.selectionEnd = l.selectionEnd;
        }
        this.updateTextareaPosition();
      }
    }, updateFromTextArea: function () {
      if (this.hiddenTextarea) {
        this.cursorOffsetCache = {}, this.text = this.hiddenTextarea.value, this._shouldClearDimensionCache() && (this.initDimensions(), this.setCoords());
        var l = this.fromStringToGraphemeSelection(this.hiddenTextarea.selectionStart, this.hiddenTextarea.selectionEnd, this.hiddenTextarea.value);
        this.selectionEnd = this.selectionStart = l.selectionEnd, this.inCompositionMode || (this.selectionStart = l.selectionStart), this.updateTextareaPosition();
      }
    }, updateTextareaPosition: function () {
      if (this.selectionStart === this.selectionEnd) {
        var l = this._calcTextareaPosition();
        this.hiddenTextarea.style.left = l.left, this.hiddenTextarea.style.top = l.top;
      }
    }, _calcTextareaPosition: function () {
      if (!this.canvas) return { x: 1, y: 1 };
      var l = this.inCompositionMode ? this.compositionStart : this.selectionStart, a = this._getCursorBoundaries(l), c = this.get2DCursorLocation(l), f = c.lineIndex, i = c.charIndex, r = this.getValueOfPropertyAt(f, i, "fontSize") * this.lineHeight, s = a.leftOffset, u = this.calcTransformMatrix(), h = { x: a.left + s, y: a.top + a.topOffset + r }, p = this.canvas.getRetinaScaling(), m = this.canvas.upperCanvasEl, y = m.width / p, x = m.height / p, S = y - r, k = x - r, D = m.clientWidth / y, M = m.clientHeight / x;
      return h = v.util.transformPoint(h, u), (h = v.util.transformPoint(h, this.canvas.viewportTransform)).x *= D, h.y *= M, h.x < 0 && (h.x = 0), h.x > S && (h.x = S), h.y < 0 && (h.y = 0), h.y > k && (h.y = k), h.x += this.canvas._offset.left, h.y += this.canvas._offset.top, { left: h.x + "px", top: h.y + "px", fontSize: r + "px", charHeight: r };
    }, _saveEditingProps: function () {
      this._savedProps = { hasControls: this.hasControls, borderColor: this.borderColor, lockMovementX: this.lockMovementX, lockMovementY: this.lockMovementY, hoverCursor: this.hoverCursor, selectable: this.selectable, defaultCursor: this.canvas && this.canvas.defaultCursor, moveCursor: this.canvas && this.canvas.moveCursor };
    }, _restoreEditingProps: function () {
      this._savedProps && (this.hoverCursor = this._savedProps.hoverCursor, this.hasControls = this._savedProps.hasControls, this.borderColor = this._savedProps.borderColor, this.selectable = this._savedProps.selectable, this.lockMovementX = this._savedProps.lockMovementX, this.lockMovementY = this._savedProps.lockMovementY, this.canvas && (this.canvas.defaultCursor = this._savedProps.defaultCursor, this.canvas.moveCursor = this._savedProps.moveCursor));
    }, exitEditing: function () {
      var l = this._textBeforeEdit !== this.text, a = this.hiddenTextarea;
      return this.selected = !1, this.isEditing = !1, this.selectionEnd = this.selectionStart, a && (a.blur && a.blur(), a.parentNode && a.parentNode.removeChild(a)), this.hiddenTextarea = null, this.abortCursorAnimation(), this._restoreEditingProps(), this._currentCursorOpacity = 0, this._shouldClearDimensionCache() && (this.initDimensions(), this.setCoords()), this.fire("editing:exited"), l && this.fire("modified"), this.canvas && (this.canvas.off("mouse:move", this.mouseMoveHandler), this.canvas.fire("text:editing:exited", { target: this }), l && this.canvas.fire("object:modified", { target: this })), this;
    }, _removeExtraneousStyles: function () {
      for (var l in this.styles) this._textLines[l] || delete this.styles[l];
    }, removeStyleFromTo: function (l, a) {
      var c, f, i = this.get2DCursorLocation(l, !0), r = this.get2DCursorLocation(a, !0), s = i.lineIndex, u = i.charIndex, h = r.lineIndex, p = r.charIndex;
      if (s !== h) {
        if (this.styles[s]) for (c = u; c < this._unwrappedTextLines[s].length; c++) delete this.styles[s][c];
        if (this.styles[h]) for (c = p; c < this._unwrappedTextLines[h].length; c++) (f = this.styles[h][c]) && (this.styles[s] || (this.styles[s] = {}), this.styles[s][u + c - p] = f);
        for (c = s + 1; c <= h; c++) delete this.styles[c];
        this.shiftLineStyles(h, s - h);
      } else if (this.styles[s]) {
        f = this.styles[s];
        var m, y, x = p - u;
        for (c = u; c < p; c++) delete f[c];
        for (y in this.styles[s]) (m = parseInt(y, 10)) >= p && (f[m - x] = f[y], delete f[y]);
      }
    }, shiftLineStyles: function (l, a) {
      var c = ht(this.styles);
      for (var f in this.styles) {
        var i = parseInt(f, 10);
        i > l && (this.styles[i + a] = c[i], c[i - a] || delete this.styles[i]);
      }
    }, restartCursorIfNeeded: function () {
      this._currentTickState && !this._currentTickState.isAborted && this._currentTickCompleteState && !this._currentTickCompleteState.isAborted || this.initDelayedCursor();
    }, insertNewlineStyleObject: function (l, a, c, f) {
      var i, r = {}, s = !1, u = this._unwrappedTextLines[l].length === a;
      for (var h in c || (c = 1), this.shiftLineStyles(l, c), this.styles[l] && (i = this.styles[l][a === 0 ? a : a - 1]), this.styles[l]) {
        var p = parseInt(h, 10);
        p >= a && (s = !0, r[p - a] = this.styles[l][h], u && a === 0 || delete this.styles[l][h]);
      }
      var m = !1;
      for (s && !u && (this.styles[l + c] = r, m = !0), m && c--; c > 0;) f && f[c - 1] ? this.styles[l + c] = { 0: ht(f[c - 1]) } : i ? this.styles[l + c] = { 0: ht(i) } : delete this.styles[l + c], c--;
      this._forceClearCache = !0;
    }, insertCharStyleObject: function (l, a, c, f) {
      this.styles || (this.styles = {});
      var i = this.styles[l], r = i ? ht(i) : {};
      for (var s in c || (c = 1), r) {
        var u = parseInt(s, 10);
        u >= a && (i[u + c] = r[u], r[u - c] || delete i[u]);
      }
      if (this._forceClearCache = !0, f) for (; c--;) Object.keys(f[c]).length && (this.styles[l] || (this.styles[l] = {}), this.styles[l][a + c] = ht(f[c]));
      else if (i) for (var h = i[a ? a - 1 : 1]; h && c--;) this.styles[l][a + c] = ht(h);
    }, insertNewStyleBlock: function (l, a, c) {
      for (var f = this.get2DCursorLocation(a, !0), i = [0], r = 0, s = 0; s < l.length; s++) l[s] === `
` ? i[++r] = 0 : i[r]++;
      for (i[0] > 0 && (this.insertCharStyleObject(f.lineIndex, f.charIndex, i[0], c), c = c && c.slice(i[0] + 1)), r && this.insertNewlineStyleObject(f.lineIndex, f.charIndex + i[0], r), s = 1; s < r; s++) i[s] > 0 ? this.insertCharStyleObject(f.lineIndex + s, 0, i[s], c) : c && (this.styles[f.lineIndex + s][0] = c[0]), c = c && c.slice(i[s] + 1);
      i[s] > 0 && this.insertCharStyleObject(f.lineIndex + s, 0, i[s], c);
    }, setSelectionStartEndWithShift: function (l, a, c) {
      c <= l ? (a === l ? this._selectionDirection = "left" : this._selectionDirection === "right" && (this._selectionDirection = "left", this.selectionEnd = l), this.selectionStart = c) : c > l && c < a ? this._selectionDirection === "right" ? this.selectionEnd = c : this.selectionStart = c : (a === l ? this._selectionDirection = "right" : this._selectionDirection === "left" && (this._selectionDirection = "right", this.selectionStart = a), this.selectionEnd = c);
    }, setSelectionInBoundaries: function () {
      var l = this.text.length;
      this.selectionStart > l ? this.selectionStart = l : this.selectionStart < 0 && (this.selectionStart = 0), this.selectionEnd > l ? this.selectionEnd = l : this.selectionEnd < 0 && (this.selectionEnd = 0);
    }
  }), v.util.object.extend(v.IText.prototype, {
    initDoubleClickSimulation: function () {
      this.__lastClickTime = +/* @__PURE__ */ new Date(), this.__lastLastClickTime = +/* @__PURE__ */ new Date(), this.__lastPointer = {}, this.on("mousedown", this.onMouseDown);
    }, onMouseDown: function (l) {
      if (this.canvas) {
        this.__newClickTime = +/* @__PURE__ */ new Date();
        var a = l.pointer;
        this.isTripleClick(a) && (this.fire("tripleclick", l), this._stopEvent(l.e)), this.__lastLastClickTime = this.__lastClickTime, this.__lastClickTime = this.__newClickTime, this.__lastPointer = a, this.__lastIsEditing = this.isEditing, this.__lastSelected = this.selected;
      }
    }, isTripleClick: function (l) {
      return this.__newClickTime - this.__lastClickTime < 500 && this.__lastClickTime - this.__lastLastClickTime < 500 && this.__lastPointer.x === l.x && this.__lastPointer.y === l.y;
    }, _stopEvent: function (l) {
      l.preventDefault && l.preventDefault(), l.stopPropagation && l.stopPropagation();
    }, initCursorSelectionHandlers: function () {
      this.initMousedownHandler(), this.initMouseupHandler(), this.initClicks();
    }, doubleClickHandler: function (l) {
      this.isEditing && this.selectWord(this.getSelectionStartFromPointer(l.e));
    }, tripleClickHandler: function (l) {
      this.isEditing && this.selectLine(this.getSelectionStartFromPointer(l.e));
    }, initClicks: function () {
      this.on("mousedblclick", this.doubleClickHandler), this.on("tripleclick", this.tripleClickHandler);
    }, _mouseDownHandler: function (l) {
      !this.canvas || !this.editable || l.e.button && l.e.button !== 1 || (this.__isMousedown = !0, this.selected && (this.inCompositionMode = !1, this.setCursorByClick(l.e)), this.isEditing && (this.__selectionStartOnMouseDown = this.selectionStart, this.selectionStart === this.selectionEnd && this.abortCursorAnimation(), this.renderCursorOrSelection()));
    }, _mouseDownHandlerBefore: function (l) {
      !this.canvas || !this.editable || l.e.button && l.e.button !== 1 || (this.selected = this === this.canvas._activeObject);
    }, initMousedownHandler: function () {
      this.on("mousedown", this._mouseDownHandler), this.on("mousedown:before", this._mouseDownHandlerBefore);
    }, initMouseupHandler: function () {
      this.on("mouseup", this.mouseUpHandler);
    }, mouseUpHandler: function (l) {
      if (this.__isMousedown = !1, !(!this.editable || this.group || l.transform && l.transform.actionPerformed || l.e.button && l.e.button !== 1)) {
        if (this.canvas) {
          var a = this.canvas._activeObject;
          if (a && a !== this) return;
        }
        this.__lastSelected && !this.__corner ? (this.selected = !1, this.__lastSelected = !1, this.enterEditing(l.e), this.selectionStart === this.selectionEnd ? this.initDelayedCursor(!0) : this.renderCursorOrSelection()) : this.selected = !0;
      }
    }, setCursorByClick: function (l) {
      var a = this.getSelectionStartFromPointer(l), c = this.selectionStart, f = this.selectionEnd;
      l.shiftKey ? this.setSelectionStartEndWithShift(c, f, a) : (this.selectionStart = a, this.selectionEnd = a), this.isEditing && (this._fireSelectionChanged(), this._updateTextarea());
    }, getSelectionStartFromPointer: function (l) {
      for (var a, c = this.getLocalPointer(l), f = 0, i = 0, r = 0, s = 0, u = 0, h = 0, p = this._textLines.length; h < p && r <= c.y; h++) r += this.getHeightOfLine(h) * this.scaleY, u = h, h > 0 && (s += this._textLines[h - 1].length + this.missingNewlineOffset(h - 1));
      i = this._getLineLeftOffset(u) * this.scaleX, a = this._textLines[u], this.direction === "rtl" && (c.x = this.width * this.scaleX - c.x + i);
      for (var m = 0, y = a.length; m < y && (f = i, (i += this.__charBounds[u][m].kernedWidth * this.scaleX) <= c.x); m++) s++;
      return this._getNewSelectionStartFromOffset(c, f, i, s, y);
    }, _getNewSelectionStartFromOffset: function (l, a, c, f, i) {
      var r = l.x - a, s = c - l.x, u = f + (s > r || s < 0 ? 0 : 1);
      return this.flipX && (u = i - u), u > this._text.length && (u = this._text.length), u;
    }
  }), v.util.object.extend(v.IText.prototype, {
    initHiddenTextarea: function () {
      this.hiddenTextarea = v.document.createElement("textarea"), this.hiddenTextarea.setAttribute("autocapitalize", "off"), this.hiddenTextarea.setAttribute("autocorrect", "off"), this.hiddenTextarea.setAttribute("autocomplete", "off"), this.hiddenTextarea.setAttribute("spellcheck", "false"), this.hiddenTextarea.setAttribute("data-fabric-hiddentextarea", ""), this.hiddenTextarea.setAttribute("wrap", "off");
      var l = this._calcTextareaPosition();
      this.hiddenTextarea.style.cssText = "position: absolute; top: " + l.top + "; left: " + l.left + "; z-index: -999; opacity: 0; width: 1px; height: 1px; font-size: 1px; paddingｰtop: " + l.fontSize + ";", this.hiddenTextareaContainer ? this.hiddenTextareaContainer.appendChild(this.hiddenTextarea) : v.document.body.appendChild(this.hiddenTextarea), v.util.addListener(this.hiddenTextarea, "keydown", this.onKeyDown.bind(this)), v.util.addListener(this.hiddenTextarea, "keyup", this.onKeyUp.bind(this)), v.util.addListener(this.hiddenTextarea, "input", this.onInput.bind(this)), v.util.addListener(this.hiddenTextarea, "copy", this.copy.bind(this)), v.util.addListener(this.hiddenTextarea, "cut", this.copy.bind(this)), v.util.addListener(this.hiddenTextarea, "paste", this.paste.bind(this)), v.util.addListener(this.hiddenTextarea, "compositionstart", this.onCompositionStart.bind(this)), v.util.addListener(this.hiddenTextarea, "compositionupdate", this.onCompositionUpdate.bind(this)), v.util.addListener(this.hiddenTextarea, "compositionend", this.onCompositionEnd.bind(this)), !this._clickHandlerInitialized && this.canvas && (v.util.addListener(this.canvas.upperCanvasEl, "click", this.onClick.bind(this)), this._clickHandlerInitialized = !0);
    }, keysMap: { 9: "exitEditing", 27: "exitEditing", 33: "moveCursorUp", 34: "moveCursorDown", 35: "moveCursorRight", 36: "moveCursorLeft", 37: "moveCursorLeft", 38: "moveCursorUp", 39: "moveCursorRight", 40: "moveCursorDown" }, keysMapRtl: { 9: "exitEditing", 27: "exitEditing", 33: "moveCursorUp", 34: "moveCursorDown", 35: "moveCursorLeft", 36: "moveCursorRight", 37: "moveCursorRight", 38: "moveCursorUp", 39: "moveCursorLeft", 40: "moveCursorDown" }, ctrlKeysMapUp: { 67: "copy", 88: "cut" }, ctrlKeysMapDown: { 65: "selectAll" }, onClick: function () {
      this.hiddenTextarea && this.hiddenTextarea.focus();
    }, onKeyDown: function (l) {
      if (this.isEditing) {
        var a = this.direction === "rtl" ? this.keysMapRtl : this.keysMap;
        if (l.keyCode in a) this[a[l.keyCode]](l);
        else {
          if (!(l.keyCode in this.ctrlKeysMapDown) || !l.ctrlKey && !l.metaKey) return;
          this[this.ctrlKeysMapDown[l.keyCode]](l);
        }
        l.stopImmediatePropagation(), l.preventDefault(), l.keyCode >= 33 && l.keyCode <= 40 ? (this.inCompositionMode = !1, this.clearContextTop(), this.renderCursorOrSelection()) : this.canvas && this.canvas.requestRenderAll();
      }
    }, onKeyUp: function (l) {
      !this.isEditing || this._copyDone || this.inCompositionMode ? this._copyDone = !1 : l.keyCode in this.ctrlKeysMapUp && (l.ctrlKey || l.metaKey) && (this[this.ctrlKeysMapUp[l.keyCode]](l), l.stopImmediatePropagation(), l.preventDefault(), this.canvas && this.canvas.requestRenderAll());
    }, onInput: function (l) {
      var a = this.fromPaste;
      if (this.fromPaste = !1, l && l.stopPropagation(), this.isEditing) {
        var c, f, i, r, s, u = this._splitTextIntoLines(this.hiddenTextarea.value).graphemeText, h = this._text.length, p = u.length, m = p - h, y = this.selectionStart, x = this.selectionEnd, S = y !== x;
        if (this.hiddenTextarea.value === "") return this.styles = {}, this.updateFromTextArea(), this.fire("changed"), void (this.canvas && (this.canvas.fire("text:changed", { target: this }), this.canvas.requestRenderAll()));
        var k = this.fromStringToGraphemeSelection(this.hiddenTextarea.selectionStart, this.hiddenTextarea.selectionEnd, this.hiddenTextarea.value), D = y > k.selectionStart;
        S ? (c = this._text.slice(y, x), m += x - y) : p < h && (c = D ? this._text.slice(x + m, x) : this._text.slice(y, y - m)), f = u.slice(k.selectionEnd - m, k.selectionEnd), c && c.length && (f.length && (i = this.getSelectionStyles(y, y + 1, !1), i = f.map(function () {
          return i[0];
        })), S ? (r = y, s = x) : D ? (r = x - c.length, s = x) : (r = x, s = x + c.length), this.removeStyleFromTo(r, s)), f.length && (a && f.join("") === v.copiedText && !v.disableStyleCopyPaste && (i = v.copiedTextStyle), this.insertNewStyleBlock(f, y, i)), this.updateFromTextArea(), this.fire("changed"), this.canvas && (this.canvas.fire("text:changed", { target: this }), this.canvas.requestRenderAll());
      }
    }, onCompositionStart: function () {
      this.inCompositionMode = !0;
    }, onCompositionEnd: function () {
      this.inCompositionMode = !1;
    }, onCompositionUpdate: function (l) {
      this.compositionStart = l.target.selectionStart, this.compositionEnd = l.target.selectionEnd, this.updateTextareaPosition();
    }, copy: function () {
      this.selectionStart !== this.selectionEnd && (v.copiedText = this.getSelectedText(), v.disableStyleCopyPaste ? v.copiedTextStyle = null : v.copiedTextStyle = this.getSelectionStyles(this.selectionStart, this.selectionEnd, !0), this._copyDone = !0);
    }, paste: function () {
      this.fromPaste = !0;
    }, _getClipboardData: function (l) {
      return l && l.clipboardData || v.window.clipboardData;
    }, _getWidthBeforeCursor: function (l, a) {
      var c, f = this._getLineLeftOffset(l);
      return a > 0 && (f += (c = this.__charBounds[l][a - 1]).left + c.width), f;
    }, getDownCursorOffset: function (l, a) {
      var c = this._getSelectionForOffset(l, a), f = this.get2DCursorLocation(c), i = f.lineIndex;
      if (i === this._textLines.length - 1 || l.metaKey || l.keyCode === 34) return this._text.length - c;
      var r = f.charIndex, s = this._getWidthBeforeCursor(i, r), u = this._getIndexOnLine(i + 1, s);
      return this._textLines[i].slice(r).length + u + 1 + this.missingNewlineOffset(i);
    }, _getSelectionForOffset: function (l, a) {
      return l.shiftKey && this.selectionStart !== this.selectionEnd && a ? this.selectionEnd : this.selectionStart;
    }, getUpCursorOffset: function (l, a) {
      var c = this._getSelectionForOffset(l, a), f = this.get2DCursorLocation(c), i = f.lineIndex;
      if (i === 0 || l.metaKey || l.keyCode === 33) return -c;
      var r = f.charIndex, s = this._getWidthBeforeCursor(i, r), u = this._getIndexOnLine(i - 1, s), h = this._textLines[i].slice(0, r), p = this.missingNewlineOffset(i - 1);
      return -this._textLines[i - 1].length + u - h.length + (1 - p);
    }, _getIndexOnLine: function (l, a) {
      for (var c, f, i = this._textLines[l], r = this._getLineLeftOffset(l), s = 0, u = 0, h = i.length; u < h; u++) if ((r += c = this.__charBounds[l][u].width) > a) {
        f = !0;
        var p = r - c, m = r, y = Math.abs(p - a);
        s = Math.abs(m - a) < y ? u : u - 1;
        break;
      }
      return f || (s = i.length - 1), s;
    }, moveCursorDown: function (l) {
      this.selectionStart >= this._text.length && this.selectionEnd >= this._text.length || this._moveCursorUpOrDown("Down", l);
    }, moveCursorUp: function (l) {
      this.selectionStart === 0 && this.selectionEnd === 0 || this._moveCursorUpOrDown("Up", l);
    }, _moveCursorUpOrDown: function (l, a) {
      var c = this["get" + l + "CursorOffset"](a, this._selectionDirection === "right");
      a.shiftKey ? this.moveCursorWithShift(c) : this.moveCursorWithoutShift(c), c !== 0 && (this.setSelectionInBoundaries(), this.abortCursorAnimation(), this._currentCursorOpacity = 1, this.initDelayedCursor(), this._fireSelectionChanged(), this._updateTextarea());
    }, moveCursorWithShift: function (l) {
      var a = this._selectionDirection === "left" ? this.selectionStart + l : this.selectionEnd + l;
      return this.setSelectionStartEndWithShift(this.selectionStart, this.selectionEnd, a), l !== 0;
    }, moveCursorWithoutShift: function (l) {
      return l < 0 ? (this.selectionStart += l, this.selectionEnd = this.selectionStart) : (this.selectionEnd += l, this.selectionStart = this.selectionEnd), l !== 0;
    }, moveCursorLeft: function (l) {
      this.selectionStart === 0 && this.selectionEnd === 0 || this._moveCursorLeftOrRight("Left", l);
    }, _move: function (l, a, c) {
      var f;
      if (l.altKey) f = this["findWordBoundary" + c](this[a]);
      else {
        if (!l.metaKey && l.keyCode !== 35 && l.keyCode !== 36) return this[a] += c === "Left" ? -1 : 1, !0;
        f = this["findLineBoundary" + c](this[a]);
      }
      if (typeof f !== void 0 && this[a] !== f) return this[a] = f, !0;
    }, _moveLeft: function (l, a) {
      return this._move(l, a, "Left");
    }, _moveRight: function (l, a) {
      return this._move(l, a, "Right");
    }, moveCursorLeftWithoutShift: function (l) {
      var a = !0;
      return this._selectionDirection = "left", this.selectionEnd === this.selectionStart && this.selectionStart !== 0 && (a = this._moveLeft(l, "selectionStart")), this.selectionEnd = this.selectionStart, a;
    }, moveCursorLeftWithShift: function (l) {
      return this._selectionDirection === "right" && this.selectionStart !== this.selectionEnd ? this._moveLeft(l, "selectionEnd") : this.selectionStart !== 0 ? (this._selectionDirection = "left", this._moveLeft(l, "selectionStart")) : void 0;
    }, moveCursorRight: function (l) {
      this.selectionStart >= this._text.length && this.selectionEnd >= this._text.length || this._moveCursorLeftOrRight("Right", l);
    }, _moveCursorLeftOrRight: function (l, a) {
      var c = "moveCursor" + l + "With";
      this._currentCursorOpacity = 1, a.shiftKey ? c += "Shift" : c += "outShift", this[c](a) && (this.abortCursorAnimation(), this.initDelayedCursor(), this._fireSelectionChanged(), this._updateTextarea());
    }, moveCursorRightWithShift: function (l) {
      return this._selectionDirection === "left" && this.selectionStart !== this.selectionEnd ? this._moveRight(l, "selectionStart") : this.selectionEnd !== this._text.length ? (this._selectionDirection = "right", this._moveRight(l, "selectionEnd")) : void 0;
    }, moveCursorRightWithoutShift: function (l) {
      var a = !0;
      return this._selectionDirection = "right", this.selectionStart === this.selectionEnd ? (a = this._moveRight(l, "selectionStart"), this.selectionEnd = this.selectionStart) : this.selectionStart = this.selectionEnd, a;
    }, removeChars: function (l, a) {
      a === void 0 && (a = l + 1), this.removeStyleFromTo(l, a), this._text.splice(l, a - l), this.text = this._text.join(""), this.set("dirty", !0), this._shouldClearDimensionCache() && (this.initDimensions(), this.setCoords()), this._removeExtraneousStyles();
    }, insertChars: function (l, a, c, f) {
      f === void 0 && (f = c), f > c && this.removeStyleFromTo(c, f);
      var i = v.util.string.graphemeSplit(l);
      this.insertNewStyleBlock(i, c, a), this._text = [].concat(this._text.slice(0, c), i, this._text.slice(f)), this.text = this._text.join(""), this.set("dirty", !0), this._shouldClearDimensionCache() && (this.initDimensions(), this.setCoords()), this._removeExtraneousStyles();
    }
  }), function () {
    var l = v.util.toFixed, a = /  +/g;
    v.util.object.extend(v.Text.prototype, {
      _toSVG: function () {
        var c = this._getSVGLeftTopOffsets(), f = this._getSVGTextAndBg(c.textTop, c.textLeft);
        return this._wrapSVGTextAndBg(f);
      }, toSVG: function (c) {
        return this._createBaseSVGMarkup(this._toSVG(), { reviver: c, noStyle: !0, withShadow: !0 });
      }, _getSVGLeftTopOffsets: function () {
        return { textLeft: -this.width / 2, textTop: -this.height / 2, lineTop: this.getHeightOfLine(0) };
      }, _wrapSVGTextAndBg: function (c) {
        var f = this.getSvgTextDecoration(this);
        return [c.textBgRects.join(""), '        <text xml:space="preserve" ', this.fontFamily ? 'font-family="' + this.fontFamily.replace(/"/g, "'") + '" ' : "", this.fontSize ? 'font-size="' + this.fontSize + '" ' : "", this.fontStyle ? 'font-style="' + this.fontStyle + '" ' : "", this.fontWeight ? 'font-weight="' + this.fontWeight + '" ' : "", f ? 'text-decoration="' + f + '" ' : "", 'style="', this.getSvgStyles(!0), '"', this.addPaintOrder(), " >", c.textSpans.join(""), `</text>
`];
      }, _getSVGTextAndBg: function (c, f) {
        var i, r = [], s = [], u = c;
        this._setSVGBg(s);
        for (var h = 0, p = this._textLines.length; h < p; h++) i = this._getLineLeftOffset(h), (this.textBackgroundColor || this.styleHas("textBackgroundColor", h)) && this._setSVGTextLineBg(s, h, f + i, u), this._setSVGTextLineText(r, h, f + i, u), u += this.getHeightOfLine(h);
        return { textSpans: r, textBgRects: s };
      }, _createTextCharSpan: function (c, f, i, r) {
        var s = c !== c.trim() || c.match(a), u = this.getSvgSpanStyles(f, s), h = u ? 'style="' + u + '"' : "", p = f.deltaY, m = "", y = v.Object.NUM_FRACTION_DIGITS;
        return p && (m = ' dy="' + l(p, y) + '" '), ['<tspan x="', l(i, y), '" y="', l(r, y), '" ', m, h, ">", v.util.string.escapeXml(c), "</tspan>"].join("");
      }, _setSVGTextLineText: function (c, f, i, r) {
        var s, u, h, p, m, y = this.getHeightOfLine(f), x = this.textAlign.indexOf("justify") !== -1, S = "", k = 0, D = this._textLines[f];
        r += y * (1 - this._fontSizeFraction) / this.lineHeight;
        for (var M = 0, C = D.length - 1; M <= C; M++) m = M === C || this.charSpacing, S += D[M], h = this.__charBounds[f][M], k === 0 ? (i += h.kernedWidth - h.width, k += h.width) : k += h.kernedWidth, x && !m && this._reSpaceAndTab.test(D[M]) && (m = !0), m || (s = s || this.getCompleteStyleDeclaration(f, M), u = this.getCompleteStyleDeclaration(f, M + 1), m = this._hasStyleChangedForSvg(s, u)), m && (p = this._getStyleDeclaration(f, M) || {}, c.push(this._createTextCharSpan(S, p, i, r)), S = "", s = u, i += k, k = 0);
      }, _pushTextBgRect: function (c, f, i, r, s, u) {
        var h = v.Object.NUM_FRACTION_DIGITS;
        c.push("        <rect ", this._getFillAttributes(f), ' x="', l(i, h), '" y="', l(r, h), '" width="', l(s, h), '" height="', l(u, h), `"></rect>
`);
      }, _setSVGTextLineBg: function (c, f, i, r) {
        for (var s, u, h = this._textLines[f], p = this.getHeightOfLine(f) / this.lineHeight, m = 0, y = 0, x = this.getValueOfPropertyAt(f, 0, "textBackgroundColor"), S = 0, k = h.length; S < k; S++) s = this.__charBounds[f][S], (u = this.getValueOfPropertyAt(f, S, "textBackgroundColor")) !== x ? (x && this._pushTextBgRect(c, x, i + y, r, m, p), y = s.left, m = s.width, x = u) : m += s.kernedWidth;
        u && this._pushTextBgRect(c, u, i + y, r, m, p);
      }, _getFillAttributes: function (c) {
        var f = c && typeof c == "string" ? new v.Color(c) : "";
        return f && f.getSource() && f.getAlpha() !== 1 ? 'opacity="' + f.getAlpha() + '" fill="' + f.setAlpha(1).toRgb() + '"' : 'fill="' + c + '"';
      }, _getSVGLineTopOffset: function (c) {
        for (var f, i = 0, r = 0; r < c; r++) i += this.getHeightOfLine(r);
        return f = this.getHeightOfLine(r), { lineTop: i, offset: (this._fontSizeMult - this._fontSizeFraction) * f / (this.lineHeight * this._fontSizeMult) };
      }, getSvgStyles: function (c) {
        return v.Object.prototype.getSvgStyles.call(this, c) + " white-space: pre;";
      }
    });
  }(), function (l) {
    var a = l.fabric || (l.fabric = {});
    a.Textbox = a.util.createClass(a.IText, a.Observable, {
      type: "textbox", minWidth: 20, dynamicMinWidth: 2, __cachedLines: null, lockScalingFlip: !0, noScaleCache: !1, _dimensionAffectingProps: a.Text.prototype._dimensionAffectingProps.concat("width"), _wordJoiners: /[ \t\r]/, splitByGrapheme: !1, initDimensions: function () {
        this.__skipDimension || (this.isEditing && this.initDelayedCursor(), this.clearContextTop(), this._clearCache(), this.dynamicMinWidth = 0, this._styleMap = this._generateStyleMap(this._splitText()), this.dynamicMinWidth > this.width && this._set("width", this.dynamicMinWidth), this.textAlign.indexOf("justify") !== -1 && this.enlargeSpaces(), this.height = this.calcTextHeight(), this.saveState({ propertySet: "_dimensionAffectingProps" }));
      }, _generateStyleMap: function (c) {
        for (var f = 0, i = 0, r = 0, s = {}, u = 0; u < c.graphemeLines.length; u++) c.graphemeText[r] === `
` && u > 0 ? (i = 0, r++, f++) : !this.splitByGrapheme && this._reSpaceAndTab.test(c.graphemeText[r]) && u > 0 && (i++, r++), s[u] = { line: f, offset: i }, r += c.graphemeLines[u].length, i += c.graphemeLines[u].length;
        return s;
      }, styleHas: function (c, f) {
        if (this._styleMap && !this.isWrapping) {
          var i = this._styleMap[f];
          i && (f = i.line);
        }
        return a.Text.prototype.styleHas.call(this, c, f);
      }, isEmptyStyles: function (c) {
        if (!this.styles) return !0;
        var f, i, r = 0, s = !1, u = this._styleMap[c], h = this._styleMap[c + 1];
        for (var p in u && (c = u.line, r = u.offset), h && (s = h.line === c, f = h.offset), i = c === void 0 ? this.styles : { line: this.styles[c] }) for (var m in i[p]) if (m >= r && (!s || m < f)) for (var y in i[p][m]) return !1;
        return !0;
      }, _getStyleDeclaration: function (c, f) {
        if (this._styleMap && !this.isWrapping) {
          var i = this._styleMap[c];
          if (!i) return null;
          c = i.line, f = i.offset + f;
        }
        return this.callSuper("_getStyleDeclaration", c, f);
      }, _setStyleDeclaration: function (c, f, i) {
        var r = this._styleMap[c];
        c = r.line, f = r.offset + f, this.styles[c][f] = i;
      }, _deleteStyleDeclaration: function (c, f) {
        var i = this._styleMap[c];
        c = i.line, f = i.offset + f, delete this.styles[c][f];
      }, _getLineStyle: function (c) {
        var f = this._styleMap[c];
        return !!this.styles[f.line];
      }, _setLineStyle: function (c) {
        var f = this._styleMap[c];
        this.styles[f.line] = {};
      }, _wrapText: function (c, f) {
        var i, r = [];
        for (this.isWrapping = !0, i = 0; i < c.length; i++) r = r.concat(this._wrapLine(c[i], i, f));
        return this.isWrapping = !1, r;
      }, _measureWord: function (c, f, i) {
        var r, s = 0;
        i = i || 0;
        for (var u = 0, h = c.length; u < h; u++)
          s += this._getGraphemeBox(c[u], f, u + i, r, !0).kernedWidth, r = c[u];
        return s;
      }, _wrapLine: function (c, f, i, r) {
        var s = 0, u = this.splitByGrapheme, h = [], p = [], m = u ? a.util.string.graphemeSplit(c) : c.split(this._wordJoiners), y = "", x = 0, S = u ? "" : " ", k = 0, D = 0, M = 0, C = !0, O = this._getWidthOfCharSpacing();
        r = r || 0, m.length === 0 && m.push([]), i -= r;
        for (var N = 0; N < m.length; N++) y = u ? m[N] : a.util.string.graphemeSplit(m[N]), k = this._measureWord(y, f, x), x += y.length, (s += D + k - O) > i && !C ? (h.push(p), p = [], s = k, C = !0) : s += O, C || u || p.push(S), p = p.concat(y), D = u ? 0 : this._measureWord([S], f, x), x++, C = !1, k > M && (M = k);
        return N && h.push(p), M + r > this.dynamicMinWidth && (this.dynamicMinWidth = M - O + r), h;
      }, isEndOfWrapping: function (c) {
        return !this._styleMap[c + 1] || this._styleMap[c + 1].line !== this._styleMap[c].line;
      }, missingNewlineOffset: function (c) {
        return this.splitByGrapheme ? this.isEndOfWrapping(c) ? 1 : 0 : 1;
      }, _splitTextIntoLines: function (c) {
        for (var f = a.Text.prototype._splitTextIntoLines.call(this, c), i = this._wrapText(f.lines, this.width), r = new Array(i.length), s = 0; s < i.length; s++) r[s] = i[s].join("");
        return f.lines = r, f.graphemeLines = i, f;
      }, getMinWidth: function () {
        return Math.max(this.minWidth, this.dynamicMinWidth);
      }, _removeExtraneousStyles: function () {
        var c = {};
        for (var f in this._styleMap) this._textLines[f] && (c[this._styleMap[f].line] = 1);
        for (var f in this.styles) c[f] || delete this.styles[f];
      }, toObject: function (c) {
        return this.callSuper("toObject", ["minWidth", "splitByGrapheme"].concat(c));
      }
    }), a.Textbox.fromObject = function (c, f) {
      return a.Object._fromObject("Textbox", c, f, "text");
    };
  }(t), function () {
    var l = v.controlsUtils, a = l.scaleSkewCursorStyleHandler, c = l.scaleCursorStyleHandler, f = l.scalingEqually, i = l.scalingYOrSkewingX, r = l.scalingXOrSkewingY, s = l.scaleOrSkewActionName, u = v.Object.prototype.controls;
    if (u.ml = new v.Control({ x: -0.5, y: 0, cursorStyleHandler: a, actionHandler: r, getActionName: s }), u.mr = new v.Control({ x: 0.5, y: 0, cursorStyleHandler: a, actionHandler: r, getActionName: s }), u.mb = new v.Control({ x: 0, y: 0.5, cursorStyleHandler: a, actionHandler: i, getActionName: s }), u.mt = new v.Control({ x: 0, y: -0.5, cursorStyleHandler: a, actionHandler: i, getActionName: s }), u.tl = new v.Control({ x: -0.5, y: -0.5, cursorStyleHandler: c, actionHandler: f }), u.tr = new v.Control({ x: 0.5, y: -0.5, cursorStyleHandler: c, actionHandler: f }), u.bl = new v.Control({ x: -0.5, y: 0.5, cursorStyleHandler: c, actionHandler: f }), u.br = new v.Control({ x: 0.5, y: 0.5, cursorStyleHandler: c, actionHandler: f }), u.mtr = new v.Control({ x: 0, y: -0.5, actionHandler: l.rotationWithSnapping, cursorStyleHandler: l.rotationStyleHandler, offsetY: -40, withConnection: !0, actionName: "rotate" }), v.Textbox) {
      var h = v.Textbox.prototype.controls = {};
      h.mtr = u.mtr, h.tr = u.tr, h.br = u.br, h.tl = u.tl, h.bl = u.bl, h.mt = u.mt, h.mb = u.mb, h.mr = new v.Control({ x: 0.5, y: 0, actionHandler: l.changeWidth, cursorStyleHandler: a, actionName: "resizing" }), h.ml = new v.Control({ x: -0.5, y: 0, actionHandler: l.changeWidth, cursorStyleHandler: a, actionName: "resizing" });
    }
  }(), function () {
    var l = v.Object.prototype._set, a = v.Object.prototype.render, c = v.Object.prototype.toObject, f = v.Object.prototype._createBaseSVGMarkup;
    v.util.object.extend(v.Object.prototype, {
      erasable: !0, getEraser: function () {
        return this.clipPath && this.clipPath.eraser ? this.clipPath : void 0;
      }, getClipPath: function () {
        var u = this.getEraser();
        return u ? u._objects[0].clipPath : this.clipPath;
      }, setClipPath: function (u) {
        var h = this.getEraser();
        (h ? h._objects[0] : this).set("clipPath", u), this.set("dirty", !0);
      }, _updateEraserDimensions: function (u, h) {
        var p = this.getEraser();
        if (p) {
          var m = p._objects[0], y = { width: m.width, height: m.height }, x = this._getNonTransformedDimensions(), S = v.util.object.extend({ width: x.x, height: x.y }, u);
          if (y.width === S.width && y.height === S.height) return;
          var k = new v.Point((y.width - S.width) / 2, (y.height - S.height) / 2);
          p.set(S), p.setPositionByOrigin(new v.Point(0, 0), "center", "center"), m.set(S), p.set("dirty", !0), h || p.getObjects("path").forEach(function (D) {
            D.setPositionByOrigin(D.getCenterPoint().add(k), "center", "center");
          }), this.setCoords();
        }
      }, _set: function (u, h) {
        return l.call(this, u, h), u !== "width" && u !== "height" || this._updateEraserDimensions(), this;
      }, render: function (u) {
        this._updateEraserDimensions(), a.call(this, u);
      }, toObject: function (u) {
        return c.call(this, ["erasable"].concat(u));
      }, eraserToSVG: function (u) {
        var h = this.getEraser();
        if (h) {
          var p = h._objects[0].fill;
          h._objects[0].fill = "white", h.clipPathId = "CLIPPATH_" + v.Object.__uid++;
          var m = ["<defs>", "<mask " + ['id="' + h.clipPathId + '"'].join(" ") + " >", h.toSVG(u.reviver), "</mask>", "</defs>"];
          return h._objects[0].fill = p, m.join(`
`);
        }
        return "";
      }, _createBaseSVGMarkup: function (u, h) {
        var p = this.getEraser();
        if (p) {
          var m = this.eraserToSVG(h);
          this.clipPath = null;
          var y = f.call(this, u, h);
          return this.clipPath = p, [m, y.replace(">", 'mask="url(#' + p.clipPathId + ')" >')].join(`
`);
        }
        return f.call(this, u, h);
      }
    });
    var i = v.Group.prototype._restoreObjectsState, r = v.Group.prototype.toObject, s = v.Group.prototype._getBounds;
    v.util.object.extend(v.Group.prototype, {
      _getBounds: function (u, h, p) {
        if (this.eraser) return this.width = this._objects[0].width, void (this.height = this._objects[0].height);
        s.call(this, u, h, p);
      }, _addEraserPathToObjects: function (u) {
        this._objects.forEach(function (h) {
          v.EraserBrush.prototype._addPathToObjectEraser.call(v.EraserBrush.prototype, h, u);
        });
      }, applyEraserToObjects: function () {
        var u = this;
        if (this.getEraser()) {
          var h = u.calcTransformMatrix();
          u.getEraser().clone(function (p) {
            var m = p._objects[0].clipPath;
            u.clipPath = m || void 0, p.getObjects("path").forEach(function (y) {
              var x = v.util.multiplyTransformMatrices(h, y.calcTransformMatrix());
              v.util.applyTransformToObject(y, x), m ? m.clone(function (S) {
                v.EraserBrush.prototype.applyClipPathToPath.call(v.EraserBrush.prototype, y, S, h), u._addEraserPathToObjects(y);
              }) : u._addEraserPathToObjects(y);
            });
          });
        }
      }, _restoreObjectsState: function () {
        return this.erasable === !0 && this.applyEraserToObjects(), i.call(this);
      }, toObject: function (u) {
        return r.call(this, ["eraser"].concat(u));
      }
    }), v.util.object.extend(v.Canvas.prototype, {
      isErasing: function () {
        return this.isDrawingMode && this.freeDrawingBrush && this.freeDrawingBrush.type === "eraser" && this.freeDrawingBrush._isErasing;
      }, renderAll: function () {
        if (!this.contextTopDirty || this._groupSelector || this.isDrawingMode || (this.clearContext(this.contextTop), this.contextTopDirty = !1), !this.isErasing()) {
          this.hasLostContext && this.renderTopLayer(this.contextTop);
          var u = this.contextContainer;
          return this.renderCanvas(u, this._chooseObjectsToRender()), this;
        }
        this.freeDrawingBrush._render();
      }
    }), v.EraserBrush = v.util.createClass(v.PencilBrush, {
      type: "eraser", _ready: !1, _drawOverlayOnTop: !1, _isErasing: !1, initialize: function (u) {
        this.callSuper("initialize", u), this._renderBound = this._render.bind(this), this.render = this.render.bind(this);
      }, hideObject: function (u) {
        u && (u._originalOpacity = u.opacity, u.set({ opacity: 0 }));
      }, restoreObjectVisibility: function (u) {
        u && u._originalOpacity && (u.set({ opacity: u._originalOpacity }), u._originalOpacity = void 0);
      }, _isErasable: function (u) {
        return u.erasable !== !1;
      }, prepareCanvasBackgroundForLayer: function (u) {
        if (u !== "overlay") {
          var h = this.canvas.backgroundImage, p = u === "top";
          h && this._isErasable(h) === !p && this.hideObject(h);
        }
      }, prepareCanvasOverlayForLayer: function (u) {
        var h = this.canvas, p = h.overlayImage, m = !!h.overlayColor;
        if (h.overlayColor && u !== "overlay" && (this.__overlayColor = h.overlayColor, delete h.overlayColor), u === "bottom") return this.hideObject(p), !1;
        var y = u === "top", x = p && !this._isErasable(p) || m;
        return p && this._isErasable(p) === !y && this.hideObject(p), x;
      }, restoreCanvasDrawables: function () {
        var u = this.canvas;
        this.__overlayColor && (u.overlayColor = this.__overlayColor, delete this.__overlayColor), this.restoreObjectVisibility(u.backgroundImage), this.restoreObjectVisibility(u.overlayImage);
      }, prepareCollectionTraversal: function (u) {
        var h = this;
        u.forEachObject(function (p) {
          p.forEachObject && p.erasable === "deep" ? h.prepareCollectionTraversal(p) : p.erasable && h.hideObject(p);
        });
      }, restoreCollectionTraversal: function (u) {
        var h = this;
        u.forEachObject(function (p) {
          p.forEachObject && p.erasable === "deep" ? h.restoreCollectionTraversal(p) : h.restoreObjectVisibility(p);
        });
      }, prepareCanvasObjectsForLayer: function (u) {
        u === "bottom" && this.prepareCollectionTraversal(this.canvas);
      }, restoreCanvasObjectsFromLayer: function (u) {
        u === "bottom" && this.restoreCollectionTraversal(this.canvas);
      }, prepareCanvasForLayer: function (u) {
        return this.prepareCanvasBackgroundForLayer(u), this.prepareCanvasObjectsForLayer(u), this.prepareCanvasOverlayForLayer(u);
      }, restoreCanvasFromLayer: function (u) {
        this.restoreCanvasDrawables(), this.restoreCanvasObjectsFromLayer(u);
      }, renderBottomLayer: function () {
        var u = this.canvas;
        this.prepareCanvasForLayer("bottom"), u.renderCanvas(u.getContext(), u.getObjects().filter(function (h) {
          return !h.erasable || h.forEachObject;
        })), this.restoreCanvasFromLayer("bottom");
      }, renderTopLayer: function () {
        var u = this.canvas;
        this._drawOverlayOnTop = this.prepareCanvasForLayer("top"), u.renderCanvas(u.contextTop, u.getObjects()), this.callSuper("_render"), this.restoreCanvasFromLayer("top");
      }, renderOverlay: function () {
        this.prepareCanvasForLayer("overlay");
        var u = this.canvas, h = u.contextTop;
        u._renderOverlay(h), this.restoreCanvasFromLayer("overlay");
      }, _saveAndTransform: function (u) {
        this.callSuper("_saveAndTransform", u), u.globalCompositeOperation = "destination-out";
      }, needsFullRender: function () {
        return this.callSuper("needsFullRender") || this._drawOverlayOnTop;
      }, onMouseDown: function (u, h) {
        this.canvas._isMainEvent(h.e) && (this._prepareForDrawing(u), this._captureDrawingPath(u), this._isErasing = !0, this.canvas.fire("erasing:start"), this._ready = !0, this._render());
      }, _render: function () {
        this._ready && (this.isRendering = 1, this.renderBottomLayer(), this.renderTopLayer(), this.renderOverlay(), this.isRendering = 0);
      }, render: function () {
        return !!this._isErasing && (this.isRendering ? this.isRendering = v.util.requestAnimFrame(this._renderBound) : this._render(), !0);
      }, applyClipPathToPath: function (u, h, p) {
        var m = u.calcTransformMatrix(), y = h.calcTransformMatrix(), x = v.util.multiplyTransformMatrices(v.util.invertTransform(m), p);
        return v.util.applyTransformToObject(h, v.util.multiplyTransformMatrices(x, y)), u.clipPath = h, u;
      }, clonePathWithClipPath: function (u, h, p) {
        var m = h.calcTransformMatrix(), y = h.getClipPath(), x = this;
        u.clone(function (S) {
          y.clone(function (k) {
            p(x.applyClipPathToPath(S, k, m));
          });
        });
      }, _addPathToObjectEraser: function (u, h) {
        var p, m = this;
        if (u.forEachObject && u.erasable === "deep") {
          var y = u._objects.filter(function (k) {
            return k.erasable;
          });
          y.length > 0 && u.clipPath ? this.clonePathWithClipPath(h, u, function (k) {
            y.forEach(function (D) {
              m._addPathToObjectEraser(D, k);
            });
          }) : y.length > 0 && y.forEach(function (k) {
            m._addPathToObjectEraser(k, h);
          });
        } else {
          if (u.getEraser()) p = u.clipPath;
          else {
            var x = u._getNonTransformedDimensions(), S = new v.Rect({ fill: "rgb(0,0,0)", width: x.x, height: x.y, clipPath: u.clipPath, originX: "center", originY: "center" });
            p = new v.Group([S], { eraser: !0 });
          }
          h.clone(function (k) {
            k.globalCompositeOperation = "destination-out";
            var D = v.util.multiplyTransformMatrices(v.util.invertTransform(u.calcTransformMatrix()), k.calcTransformMatrix());
            v.util.applyTransformToObject(k, D), p.addWithUpdate(k), u.set({ clipPath: p, dirty: !0 }), u.fire("erasing:end", { path: k }), u.group && Array.isArray(m.__subTargets) && m.__subTargets.push(u);
          });
        }
      }, applyEraserToCanvas: function (u) {
        var h = this.canvas, p = {};
        return ["backgroundImage", "overlayImage"].forEach(function (m) {
          var y = h[m];
          y && y.erasable && (this._addPathToObjectEraser(y, u), p[m] = y);
        }, this), p;
      }, _finalizeAndAddPath: function () {
        var u = this.canvas.contextTop, h = this.canvas;
        u.closePath(), this.decimate && (this._points = this.decimatePoints(this._points, this.decimate)), h.clearContext(h.contextTop), this._isErasing = !1;
        var p = this._points && this._points.length > 1 ? this.convertPointsToSVGPath(this._points) : null;
        if (!p || this._isEmptySVGPath(p)) return h.fire("erasing:end"), void h.requestRenderAll();
        var m = this.createPath(p);
        m.setCoords(), h.fire("before:path:created", { path: m });
        var y = this.applyEraserToCanvas(m), x = this;
        this.__subTargets = [];
        var S = [];
        h.forEachObject(function (k) {
          k.erasable && k.intersectsWithObject(m, !0, !0) && (x._addPathToObjectEraser(k, m), S.push(k));
        }), h.fire("erasing:end", { path: m, targets: S, subTargets: this.__subTargets, drawables: y }), delete this.__subTargets, h.requestRenderAll(), m.setCoords(), this._resetShadow(), h.fire("path:created", { path: m });
      }
    });
  }();
})(Zp);
var Dn = Zp.fabric;
const Hh = (t, e = 1) => {
  let n = t.match(/[^#]./g);
  return n ? `rgba(${n.map((o) => parseInt(o, 16)).join(", ")}, ${e})` : t;
}, Yv = (t) => function (e, n, o, d, g) {
  var b = this.cornerSize;
  e.save(), e.translate(n, o), e.rotate(fabric.util.degreesToRadians(g.angle)), e.drawImage(t, -b / 2, -b / 2, b, b), e.restore();
}, Vv = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1721638678524, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 6714, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M955.769524 734.466768c-0.13917 0-4.313236-8.118911-9.275247-18.036795L816.444339 456.344422c-4.962012-9.923-18.093077-18.035772-29.181623-18.030655l-107.787912 0.033769c-11.089569 0.004093-20.163225 9.080819-20.163225 20.174482l0 34.266431c0 11.089569 9.073656 20.170388 20.163225 20.181645l34.267455 0.023536c11.094686 0.002047 24.193005 8.146541 29.116131 18.085914l92.175329 186.263218c4.916986 9.943466-0.128937 18.077727-11.224646 18.077727L677.409772 735.420489c-11.094686 0-20.169365 9.075703-20.169365 20.164249l0 105.501846c0 11.089569-9.074679 20.163225-20.167319 20.163225L383.359666 881.249809c-11.093662 0-20.169365-9.073656-20.169365-20.163225L363.190301 755.585761c0-11.088546-9.074679-20.164249-20.163225-20.164249L193.657116 735.421512c-11.094686 0-16.003486-8.066723-10.918677-17.923208l96.398514-186.870038c5.087878-9.856485 18.319228-17.911952 29.408797-17.906835l33.378202 0.022513c11.095709 0.005117 20.169365-9.063423 20.169365-20.151969l0-33.967626c0-11.095709-9.073656-20.164249-20.169365-20.164249l-107.283422 0.039909c-11.094686 0-24.242124 8.112772-29.226648 18.024515L74.71926 716.616214c-4.978385 9.907651-8.933463 18.014282-8.792247 18.014282 0.14224 0 0.260943 9.074679 0.260943 20.169365l0 183.128831c0 11.094686 9.067516 20.169365 20.162202 20.169365l849.507874 0c11.096732 0 20.169365-9.074679 20.169365-20.169365L956.027397 754.629993C956.020234 743.532237 955.908694 734.466768 955.769524 734.466768L955.769524 734.466768zM326.291926 294.249651l110.674659 0L436.966585 659.44205 585.181231 659.44205 585.181231 294.249651l116.595508 0c11.097756 0 14.150278-6.79373 6.794753-15.093766L524.899286 71.90875c-7.355525-8.299013-19.308765-8.226359-26.567076 0.158612L319.322188 278.997272C312.063877 287.388383 315.20338 294.249651 326.291926 294.249651L326.291926 294.249651zM326.291926 294.249651", "p-id": 6715 })), Gv = P.forwardRef(Vv), qv = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1726464554433, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 1550, ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("style", null, `
        .small {
        font: 280px sans-serif;
        font-weight: bold;
        }
    `), /* @__PURE__ */ P.createElement("path", { d: "M32.935673 0.998051 32.935673 1023.001949 32.935673 0.998051Z", opacity: 0.3, "p-id": 1551 }), /* @__PURE__ */ P.createElement("path", { d: "M64.873294 0.998051 64.873294 1023.001949 64.873294 0.998051Z", opacity: 1, "p-id": 1552 }), /* @__PURE__ */ P.createElement("path", { d: "M96.810916 0.998051 96.810916 1023.001949 96.810916 0.998051Z", opacity: 0.3, "p-id": 1553 }), /* @__PURE__ */ P.createElement("path", { d: "M128.748538 0.998051 128.748538 1023.001949 128.748538 0.998051Z", opacity: 1, "p-id": 1554 }), /* @__PURE__ */ P.createElement("path", { d: "M160.68616 0.998051 160.68616 1023.001949 160.68616 0.998051Z", opacity: 0.3, "p-id": 1555 }), /* @__PURE__ */ P.createElement("path", { d: "M192.623782 0.998051 192.623782 1023.001949 192.623782 0.998051Z", opacity: 1, "p-id": 1556 }), /* @__PURE__ */ P.createElement("path", { d: "M224.561404 0.998051 224.561404 1023.001949 224.561404 0.998051Z", opacity: 0.3, "p-id": 1557 }), /* @__PURE__ */ P.createElement("path", { d: "M256.499025 0.998051 256.499025 1023.001949 256.499025 0.998051Z", opacity: 1, "p-id": 1558 }), /* @__PURE__ */ P.createElement("path", { d: "M288.436647 0.998051 288.436647 1023.001949 288.436647 0.998051Z", opacity: 0.3, "p-id": 1559 }), /* @__PURE__ */ P.createElement("path", { d: "M320.374269 0.998051 320.374269 1023.001949 320.374269 0.998051Z", opacity: 1, "p-id": 1560 }), /* @__PURE__ */ P.createElement("path", { d: "M352.311891 0.998051 352.311891 1023.001949 352.311891 0.998051Z", opacity: 0.3, "p-id": 1561 }), /* @__PURE__ */ P.createElement("path", { d: "M384.249513 0.998051 384.249513 1023.001949 384.249513 0.998051Z", opacity: 1, "p-id": 1562 }), /* @__PURE__ */ P.createElement("path", { d: "M416.187135 0.998051 416.187135 1023.001949 416.187135 0.998051Z", opacity: 0.3, "p-id": 1563 }), /* @__PURE__ */ P.createElement("path", { d: "M448.124756 0.998051 448.124756 1023.001949 448.124756 0.998051Z", opacity: 1, "p-id": 1564 }), /* @__PURE__ */ P.createElement("path", { d: "M480.062378 0.998051 480.062378 1023.001949 480.062378 0.998051Z", opacity: 0.3, "p-id": 1565 }), /* @__PURE__ */ P.createElement("path", { d: "M512 0.998051 512 1023.001949 512 0.998051Z", opacity: 1, "p-id": 1566 }), /* @__PURE__ */ P.createElement("path", { d: "M543.937622 0.998051 543.937622 1023.001949 543.937622 0.998051Z", opacity: 0.3, "p-id": 1567 }), /* @__PURE__ */ P.createElement("path", { d: "M575.875244 0.998051 575.875244 1023.001949 575.875244 0.998051Z", opacity: 1, "p-id": 1568 }), /* @__PURE__ */ P.createElement("path", { d: "M607.812865 0.998051 607.812865 1023.001949 607.812865 0.998051Z", opacity: 0.3, "p-id": 1569 }), /* @__PURE__ */ P.createElement("path", { d: "M639.750487 0.998051 639.750487 1023.001949 639.750487 0.998051Z", opacity: 1, "p-id": 1570 }), /* @__PURE__ */ P.createElement("path", { d: "M671.688109 0.998051 671.688109 1023.001949 671.688109 0.998051Z", opacity: 0.3, "p-id": 1571 }), /* @__PURE__ */ P.createElement("path", { d: "M703.625731 0.998051 703.625731 1023.001949 703.625731 0.998051Z", opacity: 1, "p-id": 1572 }), /* @__PURE__ */ P.createElement("path", { d: "M735.563353 0.998051 735.563353 1023.001949 735.563353 0.998051Z", opacity: 0.3, "p-id": 1573 }), /* @__PURE__ */ P.createElement("path", { d: "M767.500975 0.998051 767.500975 1023.001949 767.500975 0.998051Z", opacity: 1, "p-id": 1574 }), /* @__PURE__ */ P.createElement("path", { d: "M799.438596 0.998051 799.438596 1023.001949 799.438596 0.998051Z", opacity: 0.3, "p-id": 1575 }), /* @__PURE__ */ P.createElement("path", { d: "M831.376218 0.998051 831.376218 1023.001949 831.376218 0.998051Z", opacity: 1, "p-id": 1576 }), /* @__PURE__ */ P.createElement("path", { d: "M863.31384 0.998051 863.31384 1023.001949 863.31384 0.998051Z", opacity: 0.3, "p-id": 1577 }), /* @__PURE__ */ P.createElement("path", { d: "M895.251462 0.998051 895.251462 1023.001949 895.251462 0.998051Z", opacity: 1, "p-id": 1578 }), /* @__PURE__ */ P.createElement("path", { d: "M927.189084 0.998051 927.189084 1023.001949 927.189084 0.998051Z", opacity: 0.3, "p-id": 1579 }), /* @__PURE__ */ P.createElement("path", { d: "M959.126706 0.998051 959.126706 1023.001949 959.126706 0.998051Z", opacity: 1, "p-id": 1580 }), /* @__PURE__ */ P.createElement("path", { d: "M991.064327 0.998051 991.064327 1023.001949 991.064327 0.998051Z", opacity: 0.3, "p-id": 1581 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 32.935673 1023.001949 32.935673 0.998051 32.935673Z", opacity: 0.3, "p-id": 1582 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 64.873294 1023.001949 64.873294 0.998051 64.873294Z", opacity: 1, "p-id": 1583 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 96.810916 1023.001949 96.810916 0.998051 96.810916Z", opacity: 0.3, "p-id": 1584 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 128.748538 1023.001949 128.748538 0.998051 128.748538Z", opacity: 1, "p-id": 1585 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 160.68616 1023.001949 160.68616 0.998051 160.68616Z", opacity: 0.3, "p-id": 1586 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 192.623782 1023.001949 192.623782 0.998051 192.623782Z", opacity: 1, "p-id": 1587 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 224.561404 1023.001949 224.561404 0.998051 224.561404Z", opacity: 0.3, "p-id": 1588 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 256.499025 1023.001949 256.499025 0.998051 256.499025Z", opacity: 1, "p-id": 1589 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 288.436647 1023.001949 288.436647 0.998051 288.436647Z", opacity: 0.3, "p-id": 1590 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 320.374269 1023.001949 320.374269 0.998051 320.374269Z", opacity: 1, "p-id": 1591 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 352.311891 1023.001949 352.311891 0.998051 352.311891Z", opacity: 0.3, "p-id": 1592 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 384.249513 1023.001949 384.249513 0.998051 384.249513Z", opacity: 1, "p-id": 1593 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 416.187135 1023.001949 416.187135 0.998051 416.187135Z", opacity: 0.3, "p-id": 1594 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 448.124756 1023.001949 448.124756 0.998051 448.124756Z", opacity: 1, "p-id": 1595 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 480.062378 1023.001949 480.062378 0.998051 480.062378Z", opacity: 0.3, "p-id": 1596 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 512 1023.001949 512 0.998051 512Z", opacity: 1, "p-id": 1597 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 543.937622 1023.001949 543.937622 0.998051 543.937622Z", opacity: 0.3, "p-id": 1598 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 575.875244 1023.001949 575.875244 0.998051 575.875244Z", opacity: 1, "p-id": 1599 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 607.812865 1023.001949 607.812865 0.998051 607.812865Z", opacity: 0.3, "p-id": 1600 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 639.750487 1023.001949 639.750487 0.998051 639.750487Z", opacity: 1, "p-id": 1601 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 671.688109 1023.001949 671.688109 0.998051 671.688109Z", opacity: 0.3, "p-id": 1602 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 703.625731 1023.001949 703.625731 0.998051 703.625731Z", opacity: 1, "p-id": 1603 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 735.563353 1023.001949 735.563353 0.998051 735.563353Z", opacity: 0.3, "p-id": 1604 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 767.500975 1023.001949 767.500975 0.998051 767.500975Z", opacity: 1, "p-id": 1605 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 799.438596 1023.001949 799.438596 0.998051 799.438596Z", opacity: 0.3, "p-id": 1606 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 831.376218 1023.001949 831.376218 0.998051 831.376218Z", opacity: 1, "p-id": 1607 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 863.31384 1023.001949 863.31384 0.998051 863.31384Z", opacity: 0.3, "p-id": 1608 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 895.251462 1023.001949 895.251462 0.998051 895.251462Z", opacity: 1, "p-id": 1609 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 927.189084 1023.001949 927.189084 0.998051 927.189084Z", opacity: 0.3, "p-id": 1610 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 959.126706 1023.001949 959.126706 0.998051 959.126706Z", opacity: 1, "p-id": 1611 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 991.064327 1023.001949 991.064327 0.998051 991.064327Z", opacity: 0.3, "p-id": 1612 }), /* @__PURE__ */ P.createElement("path", { d: "M384.249513 586.520452c0 0 203.546448-122.121481 318.144624-73.8817 23.755602-36.197302 47.155899-75.189146 70.065154-114.997396-112.107041-27.466355-260.45929-2.746635-260.45929-2.746635s189.841216-113.89954 306.068211-78.247173c23.304483-42.025918 45.964226-83.478955 67.823532-122.065591-92.59515-6.255782-182.266012 8.687033-182.266012 8.687033s118.915743-71.348647 226.136327-83.435041c33.732117-54.75306 65.042963-97.689201 93.239891-118.835899-559.906433 0-894.253411 638.752437-1022.003899 1022.003899l63.875244 0 191.625731-319.376218c0 0 63.875244 63.875244 255.500975 0 45.385357-15.128452 90.770713-57.140398 135.307727-113.711906-112.396476-28.498339-263.058214-3.393372-263.058214-3.393372z", "p-id": 1613 }), /* @__PURE__ */ P.createElement("text", { x: 440, y: 960, className: "small" }, "RGB")), Zv = P.forwardRef(qv), Qv = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1726464554433, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 1550, ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M32.935673 0.998051 32.935673 1023.001949 32.935673 0.998051Z", opacity: 0.3, "p-id": 1551 }), /* @__PURE__ */ P.createElement("path", { d: "M64.873294 0.998051 64.873294 1023.001949 64.873294 0.998051Z", opacity: 1, "p-id": 1552 }), /* @__PURE__ */ P.createElement("path", { d: "M96.810916 0.998051 96.810916 1023.001949 96.810916 0.998051Z", opacity: 0.3, "p-id": 1553 }), /* @__PURE__ */ P.createElement("path", { d: "M128.748538 0.998051 128.748538 1023.001949 128.748538 0.998051Z", opacity: 1, "p-id": 1554 }), /* @__PURE__ */ P.createElement("path", { d: "M160.68616 0.998051 160.68616 1023.001949 160.68616 0.998051Z", opacity: 0.3, "p-id": 1555 }), /* @__PURE__ */ P.createElement("path", { d: "M192.623782 0.998051 192.623782 1023.001949 192.623782 0.998051Z", opacity: 1, "p-id": 1556 }), /* @__PURE__ */ P.createElement("path", { d: "M224.561404 0.998051 224.561404 1023.001949 224.561404 0.998051Z", opacity: 0.3, "p-id": 1557 }), /* @__PURE__ */ P.createElement("path", { d: "M256.499025 0.998051 256.499025 1023.001949 256.499025 0.998051Z", opacity: 1, "p-id": 1558 }), /* @__PURE__ */ P.createElement("path", { d: "M288.436647 0.998051 288.436647 1023.001949 288.436647 0.998051Z", opacity: 0.3, "p-id": 1559 }), /* @__PURE__ */ P.createElement("path", { d: "M320.374269 0.998051 320.374269 1023.001949 320.374269 0.998051Z", opacity: 1, "p-id": 1560 }), /* @__PURE__ */ P.createElement("path", { d: "M352.311891 0.998051 352.311891 1023.001949 352.311891 0.998051Z", opacity: 0.3, "p-id": 1561 }), /* @__PURE__ */ P.createElement("path", { d: "M384.249513 0.998051 384.249513 1023.001949 384.249513 0.998051Z", opacity: 1, "p-id": 1562 }), /* @__PURE__ */ P.createElement("path", { d: "M416.187135 0.998051 416.187135 1023.001949 416.187135 0.998051Z", opacity: 0.3, "p-id": 1563 }), /* @__PURE__ */ P.createElement("path", { d: "M448.124756 0.998051 448.124756 1023.001949 448.124756 0.998051Z", opacity: 1, "p-id": 1564 }), /* @__PURE__ */ P.createElement("path", { d: "M480.062378 0.998051 480.062378 1023.001949 480.062378 0.998051Z", opacity: 0.3, "p-id": 1565 }), /* @__PURE__ */ P.createElement("path", { d: "M512 0.998051 512 1023.001949 512 0.998051Z", opacity: 1, "p-id": 1566 }), /* @__PURE__ */ P.createElement("path", { d: "M543.937622 0.998051 543.937622 1023.001949 543.937622 0.998051Z", opacity: 0.3, "p-id": 1567 }), /* @__PURE__ */ P.createElement("path", { d: "M575.875244 0.998051 575.875244 1023.001949 575.875244 0.998051Z", opacity: 1, "p-id": 1568 }), /* @__PURE__ */ P.createElement("path", { d: "M607.812865 0.998051 607.812865 1023.001949 607.812865 0.998051Z", opacity: 0.3, "p-id": 1569 }), /* @__PURE__ */ P.createElement("path", { d: "M639.750487 0.998051 639.750487 1023.001949 639.750487 0.998051Z", opacity: 1, "p-id": 1570 }), /* @__PURE__ */ P.createElement("path", { d: "M671.688109 0.998051 671.688109 1023.001949 671.688109 0.998051Z", opacity: 0.3, "p-id": 1571 }), /* @__PURE__ */ P.createElement("path", { d: "M703.625731 0.998051 703.625731 1023.001949 703.625731 0.998051Z", opacity: 1, "p-id": 1572 }), /* @__PURE__ */ P.createElement("path", { d: "M735.563353 0.998051 735.563353 1023.001949 735.563353 0.998051Z", opacity: 0.3, "p-id": 1573 }), /* @__PURE__ */ P.createElement("path", { d: "M767.500975 0.998051 767.500975 1023.001949 767.500975 0.998051Z", opacity: 1, "p-id": 1574 }), /* @__PURE__ */ P.createElement("path", { d: "M799.438596 0.998051 799.438596 1023.001949 799.438596 0.998051Z", opacity: 0.3, "p-id": 1575 }), /* @__PURE__ */ P.createElement("path", { d: "M831.376218 0.998051 831.376218 1023.001949 831.376218 0.998051Z", opacity: 1, "p-id": 1576 }), /* @__PURE__ */ P.createElement("path", { d: "M863.31384 0.998051 863.31384 1023.001949 863.31384 0.998051Z", opacity: 0.3, "p-id": 1577 }), /* @__PURE__ */ P.createElement("path", { d: "M895.251462 0.998051 895.251462 1023.001949 895.251462 0.998051Z", opacity: 1, "p-id": 1578 }), /* @__PURE__ */ P.createElement("path", { d: "M927.189084 0.998051 927.189084 1023.001949 927.189084 0.998051Z", opacity: 0.3, "p-id": 1579 }), /* @__PURE__ */ P.createElement("path", { d: "M959.126706 0.998051 959.126706 1023.001949 959.126706 0.998051Z", opacity: 1, "p-id": 1580 }), /* @__PURE__ */ P.createElement("path", { d: "M991.064327 0.998051 991.064327 1023.001949 991.064327 0.998051Z", opacity: 0.3, "p-id": 1581 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 32.935673 1023.001949 32.935673 0.998051 32.935673Z", opacity: 0.3, "p-id": 1582 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 64.873294 1023.001949 64.873294 0.998051 64.873294Z", opacity: 1, "p-id": 1583 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 96.810916 1023.001949 96.810916 0.998051 96.810916Z", opacity: 0.3, "p-id": 1584 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 128.748538 1023.001949 128.748538 0.998051 128.748538Z", opacity: 1, "p-id": 1585 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 160.68616 1023.001949 160.68616 0.998051 160.68616Z", opacity: 0.3, "p-id": 1586 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 192.623782 1023.001949 192.623782 0.998051 192.623782Z", opacity: 1, "p-id": 1587 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 224.561404 1023.001949 224.561404 0.998051 224.561404Z", opacity: 0.3, "p-id": 1588 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 256.499025 1023.001949 256.499025 0.998051 256.499025Z", opacity: 1, "p-id": 1589 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 288.436647 1023.001949 288.436647 0.998051 288.436647Z", opacity: 0.3, "p-id": 1590 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 320.374269 1023.001949 320.374269 0.998051 320.374269Z", opacity: 1, "p-id": 1591 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 352.311891 1023.001949 352.311891 0.998051 352.311891Z", opacity: 0.3, "p-id": 1592 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 384.249513 1023.001949 384.249513 0.998051 384.249513Z", opacity: 1, "p-id": 1593 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 416.187135 1023.001949 416.187135 0.998051 416.187135Z", opacity: 0.3, "p-id": 1594 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 448.124756 1023.001949 448.124756 0.998051 448.124756Z", opacity: 1, "p-id": 1595 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 480.062378 1023.001949 480.062378 0.998051 480.062378Z", opacity: 0.3, "p-id": 1596 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 512 1023.001949 512 0.998051 512Z", opacity: 1, "p-id": 1597 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 543.937622 1023.001949 543.937622 0.998051 543.937622Z", opacity: 0.3, "p-id": 1598 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 575.875244 1023.001949 575.875244 0.998051 575.875244Z", opacity: 1, "p-id": 1599 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 607.812865 1023.001949 607.812865 0.998051 607.812865Z", opacity: 0.3, "p-id": 1600 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 639.750487 1023.001949 639.750487 0.998051 639.750487Z", opacity: 1, "p-id": 1601 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 671.688109 1023.001949 671.688109 0.998051 671.688109Z", opacity: 0.3, "p-id": 1602 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 703.625731 1023.001949 703.625731 0.998051 703.625731Z", opacity: 1, "p-id": 1603 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 735.563353 1023.001949 735.563353 0.998051 735.563353Z", opacity: 0.3, "p-id": 1604 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 767.500975 1023.001949 767.500975 0.998051 767.500975Z", opacity: 1, "p-id": 1605 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 799.438596 1023.001949 799.438596 0.998051 799.438596Z", opacity: 0.3, "p-id": 1606 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 831.376218 1023.001949 831.376218 0.998051 831.376218Z", opacity: 1, "p-id": 1607 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 863.31384 1023.001949 863.31384 0.998051 863.31384Z", opacity: 0.3, "p-id": 1608 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 895.251462 1023.001949 895.251462 0.998051 895.251462Z", opacity: 1, "p-id": 1609 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 927.189084 1023.001949 927.189084 0.998051 927.189084Z", opacity: 0.3, "p-id": 1610 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 959.126706 1023.001949 959.126706 0.998051 959.126706Z", opacity: 1, "p-id": 1611 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 991.064327 1023.001949 991.064327 0.998051 991.064327Z", opacity: 0.3, "p-id": 1612 }), /* @__PURE__ */ P.createElement("path", { d: "M384.249513 586.520452c0 0 203.546448-122.121481 318.144624-73.8817 23.755602-36.197302 47.155899-75.189146 70.065154-114.997396-112.107041-27.466355-260.45929-2.746635-260.45929-2.746635s189.841216-113.89954 306.068211-78.247173c23.304483-42.025918 45.964226-83.478955 67.823532-122.065591-92.59515-6.255782-182.266012 8.687033-182.266012 8.687033s118.915743-71.348647 226.136327-83.435041c33.732117-54.75306 65.042963-97.689201 93.239891-118.835899-559.906433 0-894.253411 638.752437-1022.003899 1022.003899l63.875244 0 191.625731-319.376218c0 0 63.875244 63.875244 255.500975 0 45.385357-15.128452 90.770713-57.140398 135.307727-113.711906-112.396476-28.498339-263.058214-3.393372-263.058214-3.393372z", "p-id": 1613 }), /* @__PURE__ */ P.createElement("rect", { x: 770, y: 610, width: 120, height: 360 }), /* @__PURE__ */ P.createElement("rect", { x: 650, y: 730, width: 360, height: 120 })), Kv = P.forwardRef(Qv), Jv = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1726464554433, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 1550, ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M32.935673 0.998051 32.935673 1023.001949 32.935673 0.998051Z", opacity: 0.3, "p-id": 1551 }), /* @__PURE__ */ P.createElement("path", { d: "M64.873294 0.998051 64.873294 1023.001949 64.873294 0.998051Z", opacity: 1, "p-id": 1552 }), /* @__PURE__ */ P.createElement("path", { d: "M96.810916 0.998051 96.810916 1023.001949 96.810916 0.998051Z", opacity: 0.3, "p-id": 1553 }), /* @__PURE__ */ P.createElement("path", { d: "M128.748538 0.998051 128.748538 1023.001949 128.748538 0.998051Z", opacity: 1, "p-id": 1554 }), /* @__PURE__ */ P.createElement("path", { d: "M160.68616 0.998051 160.68616 1023.001949 160.68616 0.998051Z", opacity: 0.3, "p-id": 1555 }), /* @__PURE__ */ P.createElement("path", { d: "M192.623782 0.998051 192.623782 1023.001949 192.623782 0.998051Z", opacity: 1, "p-id": 1556 }), /* @__PURE__ */ P.createElement("path", { d: "M224.561404 0.998051 224.561404 1023.001949 224.561404 0.998051Z", opacity: 0.3, "p-id": 1557 }), /* @__PURE__ */ P.createElement("path", { d: "M256.499025 0.998051 256.499025 1023.001949 256.499025 0.998051Z", opacity: 1, "p-id": 1558 }), /* @__PURE__ */ P.createElement("path", { d: "M288.436647 0.998051 288.436647 1023.001949 288.436647 0.998051Z", opacity: 0.3, "p-id": 1559 }), /* @__PURE__ */ P.createElement("path", { d: "M320.374269 0.998051 320.374269 1023.001949 320.374269 0.998051Z", opacity: 1, "p-id": 1560 }), /* @__PURE__ */ P.createElement("path", { d: "M352.311891 0.998051 352.311891 1023.001949 352.311891 0.998051Z", opacity: 0.3, "p-id": 1561 }), /* @__PURE__ */ P.createElement("path", { d: "M384.249513 0.998051 384.249513 1023.001949 384.249513 0.998051Z", opacity: 1, "p-id": 1562 }), /* @__PURE__ */ P.createElement("path", { d: "M416.187135 0.998051 416.187135 1023.001949 416.187135 0.998051Z", opacity: 0.3, "p-id": 1563 }), /* @__PURE__ */ P.createElement("path", { d: "M448.124756 0.998051 448.124756 1023.001949 448.124756 0.998051Z", opacity: 1, "p-id": 1564 }), /* @__PURE__ */ P.createElement("path", { d: "M480.062378 0.998051 480.062378 1023.001949 480.062378 0.998051Z", opacity: 0.3, "p-id": 1565 }), /* @__PURE__ */ P.createElement("path", { d: "M512 0.998051 512 1023.001949 512 0.998051Z", opacity: 1, "p-id": 1566 }), /* @__PURE__ */ P.createElement("path", { d: "M543.937622 0.998051 543.937622 1023.001949 543.937622 0.998051Z", opacity: 0.3, "p-id": 1567 }), /* @__PURE__ */ P.createElement("path", { d: "M575.875244 0.998051 575.875244 1023.001949 575.875244 0.998051Z", opacity: 1, "p-id": 1568 }), /* @__PURE__ */ P.createElement("path", { d: "M607.812865 0.998051 607.812865 1023.001949 607.812865 0.998051Z", opacity: 0.3, "p-id": 1569 }), /* @__PURE__ */ P.createElement("path", { d: "M639.750487 0.998051 639.750487 1023.001949 639.750487 0.998051Z", opacity: 1, "p-id": 1570 }), /* @__PURE__ */ P.createElement("path", { d: "M671.688109 0.998051 671.688109 1023.001949 671.688109 0.998051Z", opacity: 0.3, "p-id": 1571 }), /* @__PURE__ */ P.createElement("path", { d: "M703.625731 0.998051 703.625731 1023.001949 703.625731 0.998051Z", opacity: 1, "p-id": 1572 }), /* @__PURE__ */ P.createElement("path", { d: "M735.563353 0.998051 735.563353 1023.001949 735.563353 0.998051Z", opacity: 0.3, "p-id": 1573 }), /* @__PURE__ */ P.createElement("path", { d: "M767.500975 0.998051 767.500975 1023.001949 767.500975 0.998051Z", opacity: 1, "p-id": 1574 }), /* @__PURE__ */ P.createElement("path", { d: "M799.438596 0.998051 799.438596 1023.001949 799.438596 0.998051Z", opacity: 0.3, "p-id": 1575 }), /* @__PURE__ */ P.createElement("path", { d: "M831.376218 0.998051 831.376218 1023.001949 831.376218 0.998051Z", opacity: 1, "p-id": 1576 }), /* @__PURE__ */ P.createElement("path", { d: "M863.31384 0.998051 863.31384 1023.001949 863.31384 0.998051Z", opacity: 0.3, "p-id": 1577 }), /* @__PURE__ */ P.createElement("path", { d: "M895.251462 0.998051 895.251462 1023.001949 895.251462 0.998051Z", opacity: 1, "p-id": 1578 }), /* @__PURE__ */ P.createElement("path", { d: "M927.189084 0.998051 927.189084 1023.001949 927.189084 0.998051Z", opacity: 0.3, "p-id": 1579 }), /* @__PURE__ */ P.createElement("path", { d: "M959.126706 0.998051 959.126706 1023.001949 959.126706 0.998051Z", opacity: 1, "p-id": 1580 }), /* @__PURE__ */ P.createElement("path", { d: "M991.064327 0.998051 991.064327 1023.001949 991.064327 0.998051Z", opacity: 0.3, "p-id": 1581 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 32.935673 1023.001949 32.935673 0.998051 32.935673Z", opacity: 0.3, "p-id": 1582 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 64.873294 1023.001949 64.873294 0.998051 64.873294Z", opacity: 1, "p-id": 1583 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 96.810916 1023.001949 96.810916 0.998051 96.810916Z", opacity: 0.3, "p-id": 1584 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 128.748538 1023.001949 128.748538 0.998051 128.748538Z", opacity: 1, "p-id": 1585 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 160.68616 1023.001949 160.68616 0.998051 160.68616Z", opacity: 0.3, "p-id": 1586 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 192.623782 1023.001949 192.623782 0.998051 192.623782Z", opacity: 1, "p-id": 1587 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 224.561404 1023.001949 224.561404 0.998051 224.561404Z", opacity: 0.3, "p-id": 1588 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 256.499025 1023.001949 256.499025 0.998051 256.499025Z", opacity: 1, "p-id": 1589 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 288.436647 1023.001949 288.436647 0.998051 288.436647Z", opacity: 0.3, "p-id": 1590 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 320.374269 1023.001949 320.374269 0.998051 320.374269Z", opacity: 1, "p-id": 1591 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 352.311891 1023.001949 352.311891 0.998051 352.311891Z", opacity: 0.3, "p-id": 1592 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 384.249513 1023.001949 384.249513 0.998051 384.249513Z", opacity: 1, "p-id": 1593 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 416.187135 1023.001949 416.187135 0.998051 416.187135Z", opacity: 0.3, "p-id": 1594 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 448.124756 1023.001949 448.124756 0.998051 448.124756Z", opacity: 1, "p-id": 1595 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 480.062378 1023.001949 480.062378 0.998051 480.062378Z", opacity: 0.3, "p-id": 1596 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 512 1023.001949 512 0.998051 512Z", opacity: 1, "p-id": 1597 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 543.937622 1023.001949 543.937622 0.998051 543.937622Z", opacity: 0.3, "p-id": 1598 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 575.875244 1023.001949 575.875244 0.998051 575.875244Z", opacity: 1, "p-id": 1599 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 607.812865 1023.001949 607.812865 0.998051 607.812865Z", opacity: 0.3, "p-id": 1600 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 639.750487 1023.001949 639.750487 0.998051 639.750487Z", opacity: 1, "p-id": 1601 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 671.688109 1023.001949 671.688109 0.998051 671.688109Z", opacity: 0.3, "p-id": 1602 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 703.625731 1023.001949 703.625731 0.998051 703.625731Z", opacity: 1, "p-id": 1603 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 735.563353 1023.001949 735.563353 0.998051 735.563353Z", opacity: 0.3, "p-id": 1604 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 767.500975 1023.001949 767.500975 0.998051 767.500975Z", opacity: 1, "p-id": 1605 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 799.438596 1023.001949 799.438596 0.998051 799.438596Z", opacity: 0.3, "p-id": 1606 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 831.376218 1023.001949 831.376218 0.998051 831.376218Z", opacity: 1, "p-id": 1607 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 863.31384 1023.001949 863.31384 0.998051 863.31384Z", opacity: 0.3, "p-id": 1608 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 895.251462 1023.001949 895.251462 0.998051 895.251462Z", opacity: 1, "p-id": 1609 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 927.189084 1023.001949 927.189084 0.998051 927.189084Z", opacity: 0.3, "p-id": 1610 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 959.126706 1023.001949 959.126706 0.998051 959.126706Z", opacity: 1, "p-id": 1611 }), /* @__PURE__ */ P.createElement("path", { d: "M0.998051 991.064327 1023.001949 991.064327 0.998051 991.064327Z", opacity: 0.3, "p-id": 1612 }), /* @__PURE__ */ P.createElement("path", { d: "M384.249513 586.520452c0 0 203.546448-122.121481 318.144624-73.8817 23.755602-36.197302 47.155899-75.189146 70.065154-114.997396-112.107041-27.466355-260.45929-2.746635-260.45929-2.746635s189.841216-113.89954 306.068211-78.247173c23.304483-42.025918 45.964226-83.478955 67.823532-122.065591-92.59515-6.255782-182.266012 8.687033-182.266012 8.687033s118.915743-71.348647 226.136327-83.435041c33.732117-54.75306 65.042963-97.689201 93.239891-118.835899-559.906433 0-894.253411 638.752437-1022.003899 1022.003899l63.875244 0 191.625731-319.376218c0 0 63.875244 63.875244 255.500975 0 45.385357-15.128452 90.770713-57.140398 135.307727-113.711906-112.396476-28.498339-263.058214-3.393372-263.058214-3.393372z", "p-id": 1613 }), /* @__PURE__ */ P.createElement("rect", { x: 650, y: 730, width: 360, height: 120 })), $v = P.forwardRef(Jv), ty = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1720327293062, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 2436, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M128 938.666667c-25.6 0-42.666667-17.066667-42.666667-42.666667s17.066667-42.666667 42.666667-42.666667h132.266667l-140.8-140.8c-34.133333-34.133333-34.133333-85.333333 0-119.466666L601.6 110.933333c34.133333-34.133333 85.333333-34.133333 119.466667 0l179.2 179.2c34.133333 34.133333 34.133333 85.333333 0 119.466667L460.8 853.333333H896c25.6 0 42.666667 17.066667 42.666667 42.666667s-17.066667 42.666667-42.666667 42.666667H128z m68.266667-298.666667c-8.533333 8.533333-8.533333 21.333333 0 29.866667l106.666666 106.666666c34.133333 34.133333 85.333333 34.133333 119.466667 0l119.466667-119.466666-179.2-179.2L196.266667 640z", "p-id": 2437 })), ey = P.forwardRef(ty), ny = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1721638183200, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 1448, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M607.274667 612.992l88.661333 190.122667a21.333333 21.333333 0 0 1-10.325333 28.373333l-77.312 36.053333a21.333333 21.333333 0 0 1-28.373334-10.325333l-90.666666-194.474667-111.488 111.488A21.333333 21.333333 0 0 1 341.333333 759.168V218.88a21.333333 21.333333 0 0 1 35.669334-15.786667l397.056 360.96a21.333333 21.333333 0 0 1-12.714667 37.077334l-154.069333 11.861333z", "p-id": 1449 })), iy = P.forwardRef(ny), ry = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 576 512", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M234.7 42.7L197 56.8c-3 1.1-5 4-5 7.2s2 6.1 5 7.2l37.7 14.1L248.8 123c1.1 3 4 5 7.2 5s6.1-2 7.2-5l14.1-37.7L315 71.2c3-1.1 5-4 5-7.2s-2-6.1-5-7.2L277.3 42.7 263.2 5c-1.1-3-4-5-7.2-5s-6.1 2-7.2 5L234.7 42.7zM46.1 395.4c-18.7 18.7-18.7 49.1 0 67.9l34.6 34.6c18.7 18.7 49.1 18.7 67.9 0L529.9 116.5c18.7-18.7 18.7-49.1 0-67.9L495.3 14.1c-18.7-18.7-49.1-18.7-67.9 0L46.1 395.4zM484.6 82.6l-105 105-23.3-23.3 105-105 23.3 23.3zM7.5 117.2C3 118.9 0 123.2 0 128s3 9.1 7.5 10.8L64 160l21.2 56.5c1.7 4.5 6 7.5 10.8 7.5s9.1-3 10.8-7.5L128 160l56.5-21.2c4.5-1.7 7.5-6 7.5-10.8s-3-9.1-7.5-10.8L128 96 106.8 39.5C105.1 35 100.8 32 96 32s-9.1 3-10.8 7.5L64 96 7.5 117.2zm352 256c-4.5 1.7-7.5 6-7.5 10.8s3 9.1 7.5 10.8L416 416l21.2 56.5c1.7 4.5 6 7.5 10.8 7.5s9.1-3 10.8-7.5L480 416l56.5-21.2c4.5-1.7 7.5-6 7.5-10.8s-3-9.1-7.5-10.8L480 352l-21.2-56.5c-1.7-4.5-6-7.5-10.8-7.5s-9.1 3-10.8 7.5L416 352l-56.5 21.2z" })), ay = P.forwardRef(ry), oy = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1721704033853, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 4365, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M512 324.266667V136.533333c0-6.826667-3.413333-13.653333-10.24-13.653333-6.826667-3.413333-13.653333-3.413333-17.066667 0l-477.866666 307.2c-3.413333 3.413333-6.826667 6.826667-6.826667 13.653333s3.413333 10.24 6.826667 13.653334l477.866666 341.333333c6.826667 3.413333 13.653333 3.413333 17.066667 0 6.826667-3.413333 10.24-10.24 10.24-13.653333v-187.733334c249.173333 10.24 474.453333 235.52 477.866667 290.133334 0 10.24 6.826667 17.066667 17.066666 17.066666s17.066667-6.826667 17.066667-17.066666c-3.413333-225.28-170.666667-552.96-512-563.2z", "p-id": 4366 })), sy = P.forwardRef(oy), ly = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1721704049355, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 4547, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M1017.173333 430.08l-477.866666-307.2c-6.826667-3.413333-13.653333-3.413333-17.066667 0-6.826667 3.413333-10.24 6.826667-10.24 13.653333v187.733334C170.666667 334.506667 3.413333 662.186667 0 887.466667v3.413333c0 6.826667 6.826667 13.653333 17.066667 13.653333s17.066667-6.826667 17.066666-17.066666c3.413333-51.2 228.693333-279.893333 477.866667-290.133334V785.066667c0 6.826667 3.413333 13.653333 10.24 13.653333 6.826667 3.413333 13.653333 3.413333 17.066667 0l477.866666-341.333333c3.413333-3.413333 6.826667-10.24 6.826667-13.653334s-3.413333-10.24-6.826667-13.653333z", "p-id": 4548 })), cy = P.forwardRef(ly), uy = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1721712684369, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 1463, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M369.792 704.32L930.304 128 1024 223.616 369.984 896l-20.288-20.864-0.128 0.128L0 516.8 96.128 423.68l273.664 280.64z", "p-id": 1464 })), hy = P.forwardRef(uy), fy = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1721712695744, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 4043, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M960 154.24L869.76 64 512 421.76 154.24 64 64 154.24 421.76 512 64 869.76 154.24 960 512 602.24 869.76 960 960 869.76 602.24 512z", "p-id": 4044 })), dy = P.forwardRef(fy), py = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1722477182503, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 8793, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M360 184h-8c4.4 0 8-3.6 8-8v8h304v-8c0 4.4 3.6 8 8 8h-8v72h72v-80c0-35.3-28.7-64-64-64H352c-35.3 0-64 28.7-64 64v80h72v-72z", "p-id": 8794 }), /* @__PURE__ */ P.createElement("path", { d: "M864 256H160c-17.7 0-32 14.3-32 32v32c0 4.4 3.6 8 8 8h60.4l24.7 523c1.6 34.1 29.8 61 63.9 61h454c34.2 0 62.3-26.8 63.9-61l24.7-523H888c4.4 0 8-3.6 8-8v-32c0-17.7-14.3-32-32-32zM731.3 840H292.7l-24.2-512h487l-24.2 512z", "p-id": 8795 })), Wh = P.forwardRef(py), gy = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1722477192628, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 8976, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M942.3 486.4l-0.1-0.1-0.1-0.1c-36.4-76.7-80-138.7-130.7-186L760.7 351c43.7 40.2 81.5 93.7 114.1 160.9C791.5 684.2 673.4 766 512 766c-51.3 0-98.3-8.3-141.2-25.1l-54.7 54.7C374.6 823.8 439.8 838 512 838c192.2 0 335.4-100.5 430.2-300.3 7.7-16.2 7.7-35 0.1-51.3zM878.3 154.2l-42.4-42.4c-3.1-3.1-8.2-3.1-11.3 0L707.8 228.5C649.4 200.2 584.2 186 512 186c-192.2 0-335.4 100.5-430.2 300.3v0.1c-7.7 16.2-7.7 35.2 0 51.5 36.4 76.7 80 138.7 130.7 186.1L111.8 824.5c-3.1 3.1-3.1 8.2 0 11.3l42.4 42.4c3.1 3.1 8.2 3.1 11.3 0l712.8-712.8c3.1-3 3.1-8.1 0-11.2zM398.9 537.4c-1.9-8.2-2.9-16.7-2.9-25.4 0-61.9 50.1-112 112-112 8.7 0 17.3 1 25.4 2.9L398.9 537.4z m184.5-184.5C560.5 342.1 535 336 508 336c-97.2 0-176 78.8-176 176 0 27 6.1 52.5 16.9 75.4L263.3 673c-43.7-40.2-81.5-93.7-114.1-160.9C232.6 339.8 350.7 258 512 258c51.3 0 98.3 8.3 141.2 25.1l-69.8 69.8z", "p-id": 8977 }), /* @__PURE__ */ P.createElement("path", { d: "M508 624c-6.4 0-12.7-0.5-18.8-1.6l-51.1 51.1c21.4 9.3 45.1 14.4 69.9 14.4 97.2 0 176-78.8 176-176 0-24.8-5.1-48.5-14.4-69.9l-51.1 51.1c1 6.1 1.6 12.4 1.6 18.8C620 573.9 569.9 624 508 624z", "p-id": 8978 })), my = P.forwardRef(gy), vy = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1722154899261, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 1632, xmlnsXlink: "http://www.w3.org/1999/xlink", ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M512 74.666667c-17.066667 0-32 14.933333-32 32v149.333333c0 17.066667 14.933333 32 32 32s32-14.933333 32-32V106.666667c0-17.066667-14.933333-32-32-32zM693.333333 362.666667c8.533333 0 17.066667-2.133333 23.466667-8.533334l104.533333-104.533333c12.8-12.8 12.8-32 0-44.8-12.8-12.8-32-12.8-44.8 0l-104.533333 104.533333c-12.8 12.8-12.8 32 0 44.8 4.266667 6.4 12.8 8.533333 21.333333 8.533334zM917.333333 480h-149.333333c-17.066667 0-32 14.933333-32 32s14.933333 32 32 32h149.333333c17.066667 0 32-14.933333 32-32s-14.933333-32-32-32zM714.666667 669.866667c-12.8-12.8-32-12.8-44.8 0s-12.8 32 0 44.8l104.533333 104.533333c6.4 6.4 14.933333 8.533333 23.466667 8.533333s17.066667-2.133333 23.466666-8.533333c12.8-12.8 12.8-32 0-44.8l-106.666666-104.533333zM512 736c-17.066667 0-32 14.933333-32 32v149.333333c0 17.066667 14.933333 32 32 32s32-14.933333 32-32v-149.333333c0-17.066667-14.933333-32-32-32zM309.333333 669.866667l-104.533333 104.533333c-12.8 12.8-12.8 32 0 44.8 6.4 6.4 14.933333 8.533333 23.466667 8.533333s17.066667-2.133333 23.466666-8.533333l104.533334-104.533333c12.8-12.8 12.8-32 0-44.8s-36.266667-12.8-46.933334 0zM288 512c0-17.066667-14.933333-32-32-32H106.666667c-17.066667 0-32 14.933333-32 32s14.933333 32 32 32h149.333333c17.066667 0 32-14.933333 32-32zM247.466667 202.666667c-12.8-12.8-32-12.8-44.8 0-12.8 12.8-12.8 32 0 44.8l104.533333 104.533333c6.4 6.4 14.933333 8.533333 23.466667 8.533333s17.066667-2.133333 23.466666-8.533333c12.8-12.8 12.8-32 0-44.8l-106.666666-104.533333z", "p-id": 1633 })), yy = P.forwardRef(vy), by = ({
  title: t,
  titleId: e,
  ...n
}, o) => /* @__PURE__ */ P.createElement("svg", { t: 1732524822270, className: "icon", viewBox: "0 0 1024 1024", xmlns: "http://www.w3.org/2000/svg", "p-id": 5255, width: 200, height: 200, ref: o, "aria-labelledby": e, ...n }, t ? /* @__PURE__ */ P.createElement("title", { id: e }, t) : null, /* @__PURE__ */ P.createElement("path", { d: "M213.333333 810.666667h597.333334a42.666667 42.666667 0 0 1 0 85.333333H213.333333a42.666667 42.666667 0 0 1 0-85.333333z m341.333334-248.533334l138.368-138.325333 60.330666 60.330667L512 725.504l-241.365333-241.365333 60.330666-60.330667L469.333333 562.133333V85.333333h85.333334v476.8z", "p-id": 5256 })), xy = P.forwardRef(by);
var Qp = { exports: {} }, _y = "SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED", wy = _y, Cy = wy;
function Kp() {
}
function Jp() {
}
Jp.resetWarningCache = Kp;
var Sy = function () {
  function t(o, d, g, b, _, w) {
    if (w !== Cy) {
      var E = new Error(
        "Calling PropTypes validators directly is not supported by the `prop-types` package. Use PropTypes.checkPropTypes() to call them. Read more at http://fb.me/use-check-prop-types"
      );
      throw E.name = "Invariant Violation", E;
    }
  }
  t.isRequired = t;
  function e() {
    return t;
  }
  var n = {
    array: t,
    bigint: t,
    bool: t,
    func: t,
    number: t,
    object: t,
    string: t,
    symbol: t,
    any: t,
    arrayOf: e,
    element: t,
    elementType: t,
    instanceOf: e,
    node: t,
    objectOf: e,
    oneOf: e,
    oneOfType: e,
    shape: e,
    exact: e,
    checkPropTypes: Jp,
    resetWarningCache: Kp
  };
  return n.PropTypes = n, n;
};
Qp.exports = Sy();
var Ey = Qp.exports;
const Yt = /* @__PURE__ */ Vl(Ey);
function ji(t, e, n, o) {
  function d(g) {
    return g instanceof n ? g : new n(function (b) {
      b(g);
    });
  }
  return new (n || (n = Promise))(function (g, b) {
    function _(j) {
      try {
        E(o.next(j));
      } catch (H) {
        b(H);
      }
    }
    function w(j) {
      try {
        E(o.throw(j));
      } catch (H) {
        b(H);
      }
    }
    function E(j) {
      j.done ? g(j.value) : d(j.value).then(_, w);
    }
    E((o = o.apply(t, [])).next());
  });
}
const ky = /* @__PURE__ */ new Map([
  // https://github.com/guzzle/psr7/blob/2d9260799e713f1c475d3c5fdc3d6561ff7441b2/src/MimeType.php
  ["1km", "application/vnd.1000minds.decision-model+xml"],
  ["3dml", "text/vnd.in3d.3dml"],
  ["3ds", "image/x-3ds"],
  ["3g2", "video/3gpp2"],
  ["3gp", "video/3gp"],
  ["3gpp", "video/3gpp"],
  ["3mf", "model/3mf"],
  ["7z", "application/x-7z-compressed"],
  ["7zip", "application/x-7z-compressed"],
  ["123", "application/vnd.lotus-1-2-3"],
  ["aab", "application/x-authorware-bin"],
  ["aac", "audio/x-acc"],
  ["aam", "application/x-authorware-map"],
  ["aas", "application/x-authorware-seg"],
  ["abw", "application/x-abiword"],
  ["ac", "application/vnd.nokia.n-gage.ac+xml"],
  ["ac3", "audio/ac3"],
  ["acc", "application/vnd.americandynamics.acc"],
  ["ace", "application/x-ace-compressed"],
  ["acu", "application/vnd.acucobol"],
  ["acutc", "application/vnd.acucorp"],
  ["adp", "audio/adpcm"],
  ["aep", "application/vnd.audiograph"],
  ["afm", "application/x-font-type1"],
  ["afp", "application/vnd.ibm.modcap"],
  ["ahead", "application/vnd.ahead.space"],
  ["ai", "application/pdf"],
  ["aif", "audio/x-aiff"],
  ["aifc", "audio/x-aiff"],
  ["aiff", "audio/x-aiff"],
  ["air", "application/vnd.adobe.air-application-installer-package+zip"],
  ["ait", "application/vnd.dvb.ait"],
  ["ami", "application/vnd.amiga.ami"],
  ["amr", "audio/amr"],
  ["apk", "application/vnd.android.package-archive"],
  ["apng", "image/apng"],
  ["appcache", "text/cache-manifest"],
  ["application", "application/x-ms-application"],
  ["apr", "application/vnd.lotus-approach"],
  ["arc", "application/x-freearc"],
  ["arj", "application/x-arj"],
  ["asc", "application/pgp-signature"],
  ["asf", "video/x-ms-asf"],
  ["asm", "text/x-asm"],
  ["aso", "application/vnd.accpac.simply.aso"],
  ["asx", "video/x-ms-asf"],
  ["atc", "application/vnd.acucorp"],
  ["atom", "application/atom+xml"],
  ["atomcat", "application/atomcat+xml"],
  ["atomdeleted", "application/atomdeleted+xml"],
  ["atomsvc", "application/atomsvc+xml"],
  ["atx", "application/vnd.antix.game-component"],
  ["au", "audio/x-au"],
  ["avi", "video/x-msvideo"],
  ["avif", "image/avif"],
  ["aw", "application/applixware"],
  ["azf", "application/vnd.airzip.filesecure.azf"],
  ["azs", "application/vnd.airzip.filesecure.azs"],
  ["azv", "image/vnd.airzip.accelerator.azv"],
  ["azw", "application/vnd.amazon.ebook"],
  ["b16", "image/vnd.pco.b16"],
  ["bat", "application/x-msdownload"],
  ["bcpio", "application/x-bcpio"],
  ["bdf", "application/x-font-bdf"],
  ["bdm", "application/vnd.syncml.dm+wbxml"],
  ["bdoc", "application/x-bdoc"],
  ["bed", "application/vnd.realvnc.bed"],
  ["bh2", "application/vnd.fujitsu.oasysprs"],
  ["bin", "application/octet-stream"],
  ["blb", "application/x-blorb"],
  ["blorb", "application/x-blorb"],
  ["bmi", "application/vnd.bmi"],
  ["bmml", "application/vnd.balsamiq.bmml+xml"],
  ["bmp", "image/bmp"],
  ["book", "application/vnd.framemaker"],
  ["box", "application/vnd.previewsystems.box"],
  ["boz", "application/x-bzip2"],
  ["bpk", "application/octet-stream"],
  ["bpmn", "application/octet-stream"],
  ["bsp", "model/vnd.valve.source.compiled-map"],
  ["btif", "image/prs.btif"],
  ["buffer", "application/octet-stream"],
  ["bz", "application/x-bzip"],
  ["bz2", "application/x-bzip2"],
  ["c", "text/x-c"],
  ["c4d", "application/vnd.clonk.c4group"],
  ["c4f", "application/vnd.clonk.c4group"],
  ["c4g", "application/vnd.clonk.c4group"],
  ["c4p", "application/vnd.clonk.c4group"],
  ["c4u", "application/vnd.clonk.c4group"],
  ["c11amc", "application/vnd.cluetrust.cartomobile-config"],
  ["c11amz", "application/vnd.cluetrust.cartomobile-config-pkg"],
  ["cab", "application/vnd.ms-cab-compressed"],
  ["caf", "audio/x-caf"],
  ["cap", "application/vnd.tcpdump.pcap"],
  ["car", "application/vnd.curl.car"],
  ["cat", "application/vnd.ms-pki.seccat"],
  ["cb7", "application/x-cbr"],
  ["cba", "application/x-cbr"],
  ["cbr", "application/x-cbr"],
  ["cbt", "application/x-cbr"],
  ["cbz", "application/x-cbr"],
  ["cc", "text/x-c"],
  ["cco", "application/x-cocoa"],
  ["cct", "application/x-director"],
  ["ccxml", "application/ccxml+xml"],
  ["cdbcmsg", "application/vnd.contact.cmsg"],
  ["cda", "application/x-cdf"],
  ["cdf", "application/x-netcdf"],
  ["cdfx", "application/cdfx+xml"],
  ["cdkey", "application/vnd.mediastation.cdkey"],
  ["cdmia", "application/cdmi-capability"],
  ["cdmic", "application/cdmi-container"],
  ["cdmid", "application/cdmi-domain"],
  ["cdmio", "application/cdmi-object"],
  ["cdmiq", "application/cdmi-queue"],
  ["cdr", "application/cdr"],
  ["cdx", "chemical/x-cdx"],
  ["cdxml", "application/vnd.chemdraw+xml"],
  ["cdy", "application/vnd.cinderella"],
  ["cer", "application/pkix-cert"],
  ["cfs", "application/x-cfs-compressed"],
  ["cgm", "image/cgm"],
  ["chat", "application/x-chat"],
  ["chm", "application/vnd.ms-htmlhelp"],
  ["chrt", "application/vnd.kde.kchart"],
  ["cif", "chemical/x-cif"],
  ["cii", "application/vnd.anser-web-certificate-issue-initiation"],
  ["cil", "application/vnd.ms-artgalry"],
  ["cjs", "application/node"],
  ["cla", "application/vnd.claymore"],
  ["class", "application/octet-stream"],
  ["clkk", "application/vnd.crick.clicker.keyboard"],
  ["clkp", "application/vnd.crick.clicker.palette"],
  ["clkt", "application/vnd.crick.clicker.template"],
  ["clkw", "application/vnd.crick.clicker.wordbank"],
  ["clkx", "application/vnd.crick.clicker"],
  ["clp", "application/x-msclip"],
  ["cmc", "application/vnd.cosmocaller"],
  ["cmdf", "chemical/x-cmdf"],
  ["cml", "chemical/x-cml"],
  ["cmp", "application/vnd.yellowriver-custom-menu"],
  ["cmx", "image/x-cmx"],
  ["cod", "application/vnd.rim.cod"],
  ["coffee", "text/coffeescript"],
  ["com", "application/x-msdownload"],
  ["conf", "text/plain"],
  ["cpio", "application/x-cpio"],
  ["cpp", "text/x-c"],
  ["cpt", "application/mac-compactpro"],
  ["crd", "application/x-mscardfile"],
  ["crl", "application/pkix-crl"],
  ["crt", "application/x-x509-ca-cert"],
  ["crx", "application/x-chrome-extension"],
  ["cryptonote", "application/vnd.rig.cryptonote"],
  ["csh", "application/x-csh"],
  ["csl", "application/vnd.citationstyles.style+xml"],
  ["csml", "chemical/x-csml"],
  ["csp", "application/vnd.commonspace"],
  ["csr", "application/octet-stream"],
  ["css", "text/css"],
  ["cst", "application/x-director"],
  ["csv", "text/csv"],
  ["cu", "application/cu-seeme"],
  ["curl", "text/vnd.curl"],
  ["cww", "application/prs.cww"],
  ["cxt", "application/x-director"],
  ["cxx", "text/x-c"],
  ["dae", "model/vnd.collada+xml"],
  ["daf", "application/vnd.mobius.daf"],
  ["dart", "application/vnd.dart"],
  ["dataless", "application/vnd.fdsn.seed"],
  ["davmount", "application/davmount+xml"],
  ["dbf", "application/vnd.dbf"],
  ["dbk", "application/docbook+xml"],
  ["dcr", "application/x-director"],
  ["dcurl", "text/vnd.curl.dcurl"],
  ["dd2", "application/vnd.oma.dd2+xml"],
  ["ddd", "application/vnd.fujixerox.ddd"],
  ["ddf", "application/vnd.syncml.dmddf+xml"],
  ["dds", "image/vnd.ms-dds"],
  ["deb", "application/x-debian-package"],
  ["def", "text/plain"],
  ["deploy", "application/octet-stream"],
  ["der", "application/x-x509-ca-cert"],
  ["dfac", "application/vnd.dreamfactory"],
  ["dgc", "application/x-dgc-compressed"],
  ["dic", "text/x-c"],
  ["dir", "application/x-director"],
  ["dis", "application/vnd.mobius.dis"],
  ["disposition-notification", "message/disposition-notification"],
  ["dist", "application/octet-stream"],
  ["distz", "application/octet-stream"],
  ["djv", "image/vnd.djvu"],
  ["djvu", "image/vnd.djvu"],
  ["dll", "application/octet-stream"],
  ["dmg", "application/x-apple-diskimage"],
  ["dmn", "application/octet-stream"],
  ["dmp", "application/vnd.tcpdump.pcap"],
  ["dms", "application/octet-stream"],
  ["dna", "application/vnd.dna"],
  ["doc", "application/msword"],
  ["docm", "application/vnd.ms-word.template.macroEnabled.12"],
  ["docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
  ["dot", "application/msword"],
  ["dotm", "application/vnd.ms-word.template.macroEnabled.12"],
  ["dotx", "application/vnd.openxmlformats-officedocument.wordprocessingml.template"],
  ["dp", "application/vnd.osgi.dp"],
  ["dpg", "application/vnd.dpgraph"],
  ["dra", "audio/vnd.dra"],
  ["drle", "image/dicom-rle"],
  ["dsc", "text/prs.lines.tag"],
  ["dssc", "application/dssc+der"],
  ["dtb", "application/x-dtbook+xml"],
  ["dtd", "application/xml-dtd"],
  ["dts", "audio/vnd.dts"],
  ["dtshd", "audio/vnd.dts.hd"],
  ["dump", "application/octet-stream"],
  ["dvb", "video/vnd.dvb.file"],
  ["dvi", "application/x-dvi"],
  ["dwd", "application/atsc-dwd+xml"],
  ["dwf", "model/vnd.dwf"],
  ["dwg", "image/vnd.dwg"],
  ["dxf", "image/vnd.dxf"],
  ["dxp", "application/vnd.spotfire.dxp"],
  ["dxr", "application/x-director"],
  ["ear", "application/java-archive"],
  ["ecelp4800", "audio/vnd.nuera.ecelp4800"],
  ["ecelp7470", "audio/vnd.nuera.ecelp7470"],
  ["ecelp9600", "audio/vnd.nuera.ecelp9600"],
  ["ecma", "application/ecmascript"],
  ["edm", "application/vnd.novadigm.edm"],
  ["edx", "application/vnd.novadigm.edx"],
  ["efif", "application/vnd.picsel"],
  ["ei6", "application/vnd.pg.osasli"],
  ["elc", "application/octet-stream"],
  ["emf", "image/emf"],
  ["eml", "message/rfc822"],
  ["emma", "application/emma+xml"],
  ["emotionml", "application/emotionml+xml"],
  ["emz", "application/x-msmetafile"],
  ["eol", "audio/vnd.digital-winds"],
  ["eot", "application/vnd.ms-fontobject"],
  ["eps", "application/postscript"],
  ["epub", "application/epub+zip"],
  ["es", "application/ecmascript"],
  ["es3", "application/vnd.eszigno3+xml"],
  ["esa", "application/vnd.osgi.subsystem"],
  ["esf", "application/vnd.epson.esf"],
  ["et3", "application/vnd.eszigno3+xml"],
  ["etx", "text/x-setext"],
  ["eva", "application/x-eva"],
  ["evy", "application/x-envoy"],
  ["exe", "application/octet-stream"],
  ["exi", "application/exi"],
  ["exp", "application/express"],
  ["exr", "image/aces"],
  ["ext", "application/vnd.novadigm.ext"],
  ["ez", "application/andrew-inset"],
  ["ez2", "application/vnd.ezpix-album"],
  ["ez3", "application/vnd.ezpix-package"],
  ["f", "text/x-fortran"],
  ["f4v", "video/mp4"],
  ["f77", "text/x-fortran"],
  ["f90", "text/x-fortran"],
  ["fbs", "image/vnd.fastbidsheet"],
  ["fcdt", "application/vnd.adobe.formscentral.fcdt"],
  ["fcs", "application/vnd.isac.fcs"],
  ["fdf", "application/vnd.fdf"],
  ["fdt", "application/fdt+xml"],
  ["fe_launch", "application/vnd.denovo.fcselayout-link"],
  ["fg5", "application/vnd.fujitsu.oasysgp"],
  ["fgd", "application/x-director"],
  ["fh", "image/x-freehand"],
  ["fh4", "image/x-freehand"],
  ["fh5", "image/x-freehand"],
  ["fh7", "image/x-freehand"],
  ["fhc", "image/x-freehand"],
  ["fig", "application/x-xfig"],
  ["fits", "image/fits"],
  ["flac", "audio/x-flac"],
  ["fli", "video/x-fli"],
  ["flo", "application/vnd.micrografx.flo"],
  ["flv", "video/x-flv"],
  ["flw", "application/vnd.kde.kivio"],
  ["flx", "text/vnd.fmi.flexstor"],
  ["fly", "text/vnd.fly"],
  ["fm", "application/vnd.framemaker"],
  ["fnc", "application/vnd.frogans.fnc"],
  ["fo", "application/vnd.software602.filler.form+xml"],
  ["for", "text/x-fortran"],
  ["fpx", "image/vnd.fpx"],
  ["frame", "application/vnd.framemaker"],
  ["fsc", "application/vnd.fsc.weblaunch"],
  ["fst", "image/vnd.fst"],
  ["ftc", "application/vnd.fluxtime.clip"],
  ["fti", "application/vnd.anser-web-funds-transfer-initiation"],
  ["fvt", "video/vnd.fvt"],
  ["fxp", "application/vnd.adobe.fxp"],
  ["fxpl", "application/vnd.adobe.fxp"],
  ["fzs", "application/vnd.fuzzysheet"],
  ["g2w", "application/vnd.geoplan"],
  ["g3", "image/g3fax"],
  ["g3w", "application/vnd.geospace"],
  ["gac", "application/vnd.groove-account"],
  ["gam", "application/x-tads"],
  ["gbr", "application/rpki-ghostbusters"],
  ["gca", "application/x-gca-compressed"],
  ["gdl", "model/vnd.gdl"],
  ["gdoc", "application/vnd.google-apps.document"],
  ["geo", "application/vnd.dynageo"],
  ["geojson", "application/geo+json"],
  ["gex", "application/vnd.geometry-explorer"],
  ["ggb", "application/vnd.geogebra.file"],
  ["ggt", "application/vnd.geogebra.tool"],
  ["ghf", "application/vnd.groove-help"],
  ["gif", "image/gif"],
  ["gim", "application/vnd.groove-identity-message"],
  ["glb", "model/gltf-binary"],
  ["gltf", "model/gltf+json"],
  ["gml", "application/gml+xml"],
  ["gmx", "application/vnd.gmx"],
  ["gnumeric", "application/x-gnumeric"],
  ["gpg", "application/gpg-keys"],
  ["gph", "application/vnd.flographit"],
  ["gpx", "application/gpx+xml"],
  ["gqf", "application/vnd.grafeq"],
  ["gqs", "application/vnd.grafeq"],
  ["gram", "application/srgs"],
  ["gramps", "application/x-gramps-xml"],
  ["gre", "application/vnd.geometry-explorer"],
  ["grv", "application/vnd.groove-injector"],
  ["grxml", "application/srgs+xml"],
  ["gsf", "application/x-font-ghostscript"],
  ["gsheet", "application/vnd.google-apps.spreadsheet"],
  ["gslides", "application/vnd.google-apps.presentation"],
  ["gtar", "application/x-gtar"],
  ["gtm", "application/vnd.groove-tool-message"],
  ["gtw", "model/vnd.gtw"],
  ["gv", "text/vnd.graphviz"],
  ["gxf", "application/gxf"],
  ["gxt", "application/vnd.geonext"],
  ["gz", "application/gzip"],
  ["gzip", "application/gzip"],
  ["h", "text/x-c"],
  ["h261", "video/h261"],
  ["h263", "video/h263"],
  ["h264", "video/h264"],
  ["hal", "application/vnd.hal+xml"],
  ["hbci", "application/vnd.hbci"],
  ["hbs", "text/x-handlebars-template"],
  ["hdd", "application/x-virtualbox-hdd"],
  ["hdf", "application/x-hdf"],
  ["heic", "image/heic"],
  ["heics", "image/heic-sequence"],
  ["heif", "image/heif"],
  ["heifs", "image/heif-sequence"],
  ["hej2", "image/hej2k"],
  ["held", "application/atsc-held+xml"],
  ["hh", "text/x-c"],
  ["hjson", "application/hjson"],
  ["hlp", "application/winhlp"],
  ["hpgl", "application/vnd.hp-hpgl"],
  ["hpid", "application/vnd.hp-hpid"],
  ["hps", "application/vnd.hp-hps"],
  ["hqx", "application/mac-binhex40"],
  ["hsj2", "image/hsj2"],
  ["htc", "text/x-component"],
  ["htke", "application/vnd.kenameaapp"],
  ["htm", "text/html"],
  ["html", "text/html"],
  ["hvd", "application/vnd.yamaha.hv-dic"],
  ["hvp", "application/vnd.yamaha.hv-voice"],
  ["hvs", "application/vnd.yamaha.hv-script"],
  ["i2g", "application/vnd.intergeo"],
  ["icc", "application/vnd.iccprofile"],
  ["ice", "x-conference/x-cooltalk"],
  ["icm", "application/vnd.iccprofile"],
  ["ico", "image/x-icon"],
  ["ics", "text/calendar"],
  ["ief", "image/ief"],
  ["ifb", "text/calendar"],
  ["ifm", "application/vnd.shana.informed.formdata"],
  ["iges", "model/iges"],
  ["igl", "application/vnd.igloader"],
  ["igm", "application/vnd.insors.igm"],
  ["igs", "model/iges"],
  ["igx", "application/vnd.micrografx.igx"],
  ["iif", "application/vnd.shana.informed.interchange"],
  ["img", "application/octet-stream"],
  ["imp", "application/vnd.accpac.simply.imp"],
  ["ims", "application/vnd.ms-ims"],
  ["in", "text/plain"],
  ["ini", "text/plain"],
  ["ink", "application/inkml+xml"],
  ["inkml", "application/inkml+xml"],
  ["install", "application/x-install-instructions"],
  ["iota", "application/vnd.astraea-software.iota"],
  ["ipfix", "application/ipfix"],
  ["ipk", "application/vnd.shana.informed.package"],
  ["irm", "application/vnd.ibm.rights-management"],
  ["irp", "application/vnd.irepository.package+xml"],
  ["iso", "application/x-iso9660-image"],
  ["itp", "application/vnd.shana.informed.formtemplate"],
  ["its", "application/its+xml"],
  ["ivp", "application/vnd.immervision-ivp"],
  ["ivu", "application/vnd.immervision-ivu"],
  ["jad", "text/vnd.sun.j2me.app-descriptor"],
  ["jade", "text/jade"],
  ["jam", "application/vnd.jam"],
  ["jar", "application/java-archive"],
  ["jardiff", "application/x-java-archive-diff"],
  ["java", "text/x-java-source"],
  ["jhc", "image/jphc"],
  ["jisp", "application/vnd.jisp"],
  ["jls", "image/jls"],
  ["jlt", "application/vnd.hp-jlyt"],
  ["jng", "image/x-jng"],
  ["jnlp", "application/x-java-jnlp-file"],
  ["joda", "application/vnd.joost.joda-archive"],
  ["jp2", "image/jp2"],
  ["jpe", "image/jpeg"],
  ["jpeg", "image/jpeg"],
  ["jpf", "image/jpx"],
  ["jpg", "image/jpeg"],
  ["jpg2", "image/jp2"],
  ["jpgm", "video/jpm"],
  ["jpgv", "video/jpeg"],
  ["jph", "image/jph"],
  ["jpm", "video/jpm"],
  ["jpx", "image/jpx"],
  ["js", "application/javascript"],
  ["json", "application/json"],
  ["json5", "application/json5"],
  ["jsonld", "application/ld+json"],
  // https://jsonlines.org/
  ["jsonl", "application/jsonl"],
  ["jsonml", "application/jsonml+json"],
  ["jsx", "text/jsx"],
  ["jxr", "image/jxr"],
  ["jxra", "image/jxra"],
  ["jxrs", "image/jxrs"],
  ["jxs", "image/jxs"],
  ["jxsc", "image/jxsc"],
  ["jxsi", "image/jxsi"],
  ["jxss", "image/jxss"],
  ["kar", "audio/midi"],
  ["karbon", "application/vnd.kde.karbon"],
  ["kdb", "application/octet-stream"],
  ["kdbx", "application/x-keepass2"],
  ["key", "application/x-iwork-keynote-sffkey"],
  ["kfo", "application/vnd.kde.kformula"],
  ["kia", "application/vnd.kidspiration"],
  ["kml", "application/vnd.google-earth.kml+xml"],
  ["kmz", "application/vnd.google-earth.kmz"],
  ["kne", "application/vnd.kinar"],
  ["knp", "application/vnd.kinar"],
  ["kon", "application/vnd.kde.kontour"],
  ["kpr", "application/vnd.kde.kpresenter"],
  ["kpt", "application/vnd.kde.kpresenter"],
  ["kpxx", "application/vnd.ds-keypoint"],
  ["ksp", "application/vnd.kde.kspread"],
  ["ktr", "application/vnd.kahootz"],
  ["ktx", "image/ktx"],
  ["ktx2", "image/ktx2"],
  ["ktz", "application/vnd.kahootz"],
  ["kwd", "application/vnd.kde.kword"],
  ["kwt", "application/vnd.kde.kword"],
  ["lasxml", "application/vnd.las.las+xml"],
  ["latex", "application/x-latex"],
  ["lbd", "application/vnd.llamagraphics.life-balance.desktop"],
  ["lbe", "application/vnd.llamagraphics.life-balance.exchange+xml"],
  ["les", "application/vnd.hhe.lesson-player"],
  ["less", "text/less"],
  ["lgr", "application/lgr+xml"],
  ["lha", "application/octet-stream"],
  ["link66", "application/vnd.route66.link66+xml"],
  ["list", "text/plain"],
  ["list3820", "application/vnd.ibm.modcap"],
  ["listafp", "application/vnd.ibm.modcap"],
  ["litcoffee", "text/coffeescript"],
  ["lnk", "application/x-ms-shortcut"],
  ["log", "text/plain"],
  ["lostxml", "application/lost+xml"],
  ["lrf", "application/octet-stream"],
  ["lrm", "application/vnd.ms-lrm"],
  ["ltf", "application/vnd.frogans.ltf"],
  ["lua", "text/x-lua"],
  ["luac", "application/x-lua-bytecode"],
  ["lvp", "audio/vnd.lucent.voice"],
  ["lwp", "application/vnd.lotus-wordpro"],
  ["lzh", "application/octet-stream"],
  ["m1v", "video/mpeg"],
  ["m2a", "audio/mpeg"],
  ["m2v", "video/mpeg"],
  ["m3a", "audio/mpeg"],
  ["m3u", "text/plain"],
  ["m3u8", "application/vnd.apple.mpegurl"],
  ["m4a", "audio/x-m4a"],
  ["m4p", "application/mp4"],
  ["m4s", "video/iso.segment"],
  ["m4u", "application/vnd.mpegurl"],
  ["m4v", "video/x-m4v"],
  ["m13", "application/x-msmediaview"],
  ["m14", "application/x-msmediaview"],
  ["m21", "application/mp21"],
  ["ma", "application/mathematica"],
  ["mads", "application/mads+xml"],
  ["maei", "application/mmt-aei+xml"],
  ["mag", "application/vnd.ecowin.chart"],
  ["maker", "application/vnd.framemaker"],
  ["man", "text/troff"],
  ["manifest", "text/cache-manifest"],
  ["map", "application/json"],
  ["mar", "application/octet-stream"],
  ["markdown", "text/markdown"],
  ["mathml", "application/mathml+xml"],
  ["mb", "application/mathematica"],
  ["mbk", "application/vnd.mobius.mbk"],
  ["mbox", "application/mbox"],
  ["mc1", "application/vnd.medcalcdata"],
  ["mcd", "application/vnd.mcd"],
  ["mcurl", "text/vnd.curl.mcurl"],
  ["md", "text/markdown"],
  ["mdb", "application/x-msaccess"],
  ["mdi", "image/vnd.ms-modi"],
  ["mdx", "text/mdx"],
  ["me", "text/troff"],
  ["mesh", "model/mesh"],
  ["meta4", "application/metalink4+xml"],
  ["metalink", "application/metalink+xml"],
  ["mets", "application/mets+xml"],
  ["mfm", "application/vnd.mfmp"],
  ["mft", "application/rpki-manifest"],
  ["mgp", "application/vnd.osgeo.mapguide.package"],
  ["mgz", "application/vnd.proteus.magazine"],
  ["mid", "audio/midi"],
  ["midi", "audio/midi"],
  ["mie", "application/x-mie"],
  ["mif", "application/vnd.mif"],
  ["mime", "message/rfc822"],
  ["mj2", "video/mj2"],
  ["mjp2", "video/mj2"],
  ["mjs", "application/javascript"],
  ["mk3d", "video/x-matroska"],
  ["mka", "audio/x-matroska"],
  ["mkd", "text/x-markdown"],
  ["mks", "video/x-matroska"],
  ["mkv", "video/x-matroska"],
  ["mlp", "application/vnd.dolby.mlp"],
  ["mmd", "application/vnd.chipnuts.karaoke-mmd"],
  ["mmf", "application/vnd.smaf"],
  ["mml", "text/mathml"],
  ["mmr", "image/vnd.fujixerox.edmics-mmr"],
  ["mng", "video/x-mng"],
  ["mny", "application/x-msmoney"],
  ["mobi", "application/x-mobipocket-ebook"],
  ["mods", "application/mods+xml"],
  ["mov", "video/quicktime"],
  ["movie", "video/x-sgi-movie"],
  ["mp2", "audio/mpeg"],
  ["mp2a", "audio/mpeg"],
  ["mp3", "audio/mpeg"],
  ["mp4", "video/mp4"],
  ["mp4a", "audio/mp4"],
  ["mp4s", "application/mp4"],
  ["mp4v", "video/mp4"],
  ["mp21", "application/mp21"],
  ["mpc", "application/vnd.mophun.certificate"],
  ["mpd", "application/dash+xml"],
  ["mpe", "video/mpeg"],
  ["mpeg", "video/mpeg"],
  ["mpg", "video/mpeg"],
  ["mpg4", "video/mp4"],
  ["mpga", "audio/mpeg"],
  ["mpkg", "application/vnd.apple.installer+xml"],
  ["mpm", "application/vnd.blueice.multipass"],
  ["mpn", "application/vnd.mophun.application"],
  ["mpp", "application/vnd.ms-project"],
  ["mpt", "application/vnd.ms-project"],
  ["mpy", "application/vnd.ibm.minipay"],
  ["mqy", "application/vnd.mobius.mqy"],
  ["mrc", "application/marc"],
  ["mrcx", "application/marcxml+xml"],
  ["ms", "text/troff"],
  ["mscml", "application/mediaservercontrol+xml"],
  ["mseed", "application/vnd.fdsn.mseed"],
  ["mseq", "application/vnd.mseq"],
  ["msf", "application/vnd.epson.msf"],
  ["msg", "application/vnd.ms-outlook"],
  ["msh", "model/mesh"],
  ["msi", "application/x-msdownload"],
  ["msl", "application/vnd.mobius.msl"],
  ["msm", "application/octet-stream"],
  ["msp", "application/octet-stream"],
  ["msty", "application/vnd.muvee.style"],
  ["mtl", "model/mtl"],
  ["mts", "model/vnd.mts"],
  ["mus", "application/vnd.musician"],
  ["musd", "application/mmt-usd+xml"],
  ["musicxml", "application/vnd.recordare.musicxml+xml"],
  ["mvb", "application/x-msmediaview"],
  ["mvt", "application/vnd.mapbox-vector-tile"],
  ["mwf", "application/vnd.mfer"],
  ["mxf", "application/mxf"],
  ["mxl", "application/vnd.recordare.musicxml"],
  ["mxmf", "audio/mobile-xmf"],
  ["mxml", "application/xv+xml"],
  ["mxs", "application/vnd.triscape.mxs"],
  ["mxu", "video/vnd.mpegurl"],
  ["n-gage", "application/vnd.nokia.n-gage.symbian.install"],
  ["n3", "text/n3"],
  ["nb", "application/mathematica"],
  ["nbp", "application/vnd.wolfram.player"],
  ["nc", "application/x-netcdf"],
  ["ncx", "application/x-dtbncx+xml"],
  ["nfo", "text/x-nfo"],
  ["ngdat", "application/vnd.nokia.n-gage.data"],
  ["nitf", "application/vnd.nitf"],
  ["nlu", "application/vnd.neurolanguage.nlu"],
  ["nml", "application/vnd.enliven"],
  ["nnd", "application/vnd.noblenet-directory"],
  ["nns", "application/vnd.noblenet-sealer"],
  ["nnw", "application/vnd.noblenet-web"],
  ["npx", "image/vnd.net-fpx"],
  ["nq", "application/n-quads"],
  ["nsc", "application/x-conference"],
  ["nsf", "application/vnd.lotus-notes"],
  ["nt", "application/n-triples"],
  ["ntf", "application/vnd.nitf"],
  ["numbers", "application/x-iwork-numbers-sffnumbers"],
  ["nzb", "application/x-nzb"],
  ["oa2", "application/vnd.fujitsu.oasys2"],
  ["oa3", "application/vnd.fujitsu.oasys3"],
  ["oas", "application/vnd.fujitsu.oasys"],
  ["obd", "application/x-msbinder"],
  ["obgx", "application/vnd.openblox.game+xml"],
  ["obj", "model/obj"],
  ["oda", "application/oda"],
  ["odb", "application/vnd.oasis.opendocument.database"],
  ["odc", "application/vnd.oasis.opendocument.chart"],
  ["odf", "application/vnd.oasis.opendocument.formula"],
  ["odft", "application/vnd.oasis.opendocument.formula-template"],
  ["odg", "application/vnd.oasis.opendocument.graphics"],
  ["odi", "application/vnd.oasis.opendocument.image"],
  ["odm", "application/vnd.oasis.opendocument.text-master"],
  ["odp", "application/vnd.oasis.opendocument.presentation"],
  ["ods", "application/vnd.oasis.opendocument.spreadsheet"],
  ["odt", "application/vnd.oasis.opendocument.text"],
  ["oga", "audio/ogg"],
  ["ogex", "model/vnd.opengex"],
  ["ogg", "audio/ogg"],
  ["ogv", "video/ogg"],
  ["ogx", "application/ogg"],
  ["omdoc", "application/omdoc+xml"],
  ["onepkg", "application/onenote"],
  ["onetmp", "application/onenote"],
  ["onetoc", "application/onenote"],
  ["onetoc2", "application/onenote"],
  ["opf", "application/oebps-package+xml"],
  ["opml", "text/x-opml"],
  ["oprc", "application/vnd.palm"],
  ["opus", "audio/ogg"],
  ["org", "text/x-org"],
  ["osf", "application/vnd.yamaha.openscoreformat"],
  ["osfpvg", "application/vnd.yamaha.openscoreformat.osfpvg+xml"],
  ["osm", "application/vnd.openstreetmap.data+xml"],
  ["otc", "application/vnd.oasis.opendocument.chart-template"],
  ["otf", "font/otf"],
  ["otg", "application/vnd.oasis.opendocument.graphics-template"],
  ["oth", "application/vnd.oasis.opendocument.text-web"],
  ["oti", "application/vnd.oasis.opendocument.image-template"],
  ["otp", "application/vnd.oasis.opendocument.presentation-template"],
  ["ots", "application/vnd.oasis.opendocument.spreadsheet-template"],
  ["ott", "application/vnd.oasis.opendocument.text-template"],
  ["ova", "application/x-virtualbox-ova"],
  ["ovf", "application/x-virtualbox-ovf"],
  ["owl", "application/rdf+xml"],
  ["oxps", "application/oxps"],
  ["oxt", "application/vnd.openofficeorg.extension"],
  ["p", "text/x-pascal"],
  ["p7a", "application/x-pkcs7-signature"],
  ["p7b", "application/x-pkcs7-certificates"],
  ["p7c", "application/pkcs7-mime"],
  ["p7m", "application/pkcs7-mime"],
  ["p7r", "application/x-pkcs7-certreqresp"],
  ["p7s", "application/pkcs7-signature"],
  ["p8", "application/pkcs8"],
  ["p10", "application/x-pkcs10"],
  ["p12", "application/x-pkcs12"],
  ["pac", "application/x-ns-proxy-autoconfig"],
  ["pages", "application/x-iwork-pages-sffpages"],
  ["pas", "text/x-pascal"],
  ["paw", "application/vnd.pawaafile"],
  ["pbd", "application/vnd.powerbuilder6"],
  ["pbm", "image/x-portable-bitmap"],
  ["pcap", "application/vnd.tcpdump.pcap"],
  ["pcf", "application/x-font-pcf"],
  ["pcl", "application/vnd.hp-pcl"],
  ["pclxl", "application/vnd.hp-pclxl"],
  ["pct", "image/x-pict"],
  ["pcurl", "application/vnd.curl.pcurl"],
  ["pcx", "image/x-pcx"],
  ["pdb", "application/x-pilot"],
  ["pde", "text/x-processing"],
  ["pdf", "application/pdf"],
  ["pem", "application/x-x509-user-cert"],
  ["pfa", "application/x-font-type1"],
  ["pfb", "application/x-font-type1"],
  ["pfm", "application/x-font-type1"],
  ["pfr", "application/font-tdpfr"],
  ["pfx", "application/x-pkcs12"],
  ["pgm", "image/x-portable-graymap"],
  ["pgn", "application/x-chess-pgn"],
  ["pgp", "application/pgp"],
  ["php", "application/x-httpd-php"],
  ["php3", "application/x-httpd-php"],
  ["php4", "application/x-httpd-php"],
  ["phps", "application/x-httpd-php-source"],
  ["phtml", "application/x-httpd-php"],
  ["pic", "image/x-pict"],
  ["pkg", "application/octet-stream"],
  ["pki", "application/pkixcmp"],
  ["pkipath", "application/pkix-pkipath"],
  ["pkpass", "application/vnd.apple.pkpass"],
  ["pl", "application/x-perl"],
  ["plb", "application/vnd.3gpp.pic-bw-large"],
  ["plc", "application/vnd.mobius.plc"],
  ["plf", "application/vnd.pocketlearn"],
  ["pls", "application/pls+xml"],
  ["pm", "application/x-perl"],
  ["pml", "application/vnd.ctc-posml"],
  ["png", "image/png"],
  ["pnm", "image/x-portable-anymap"],
  ["portpkg", "application/vnd.macports.portpkg"],
  ["pot", "application/vnd.ms-powerpoint"],
  ["potm", "application/vnd.ms-powerpoint.presentation.macroEnabled.12"],
  ["potx", "application/vnd.openxmlformats-officedocument.presentationml.template"],
  ["ppa", "application/vnd.ms-powerpoint"],
  ["ppam", "application/vnd.ms-powerpoint.addin.macroEnabled.12"],
  ["ppd", "application/vnd.cups-ppd"],
  ["ppm", "image/x-portable-pixmap"],
  ["pps", "application/vnd.ms-powerpoint"],
  ["ppsm", "application/vnd.ms-powerpoint.slideshow.macroEnabled.12"],
  ["ppsx", "application/vnd.openxmlformats-officedocument.presentationml.slideshow"],
  ["ppt", "application/powerpoint"],
  ["pptm", "application/vnd.ms-powerpoint.presentation.macroEnabled.12"],
  ["pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"],
  ["pqa", "application/vnd.palm"],
  ["prc", "application/x-pilot"],
  ["pre", "application/vnd.lotus-freelance"],
  ["prf", "application/pics-rules"],
  ["provx", "application/provenance+xml"],
  ["ps", "application/postscript"],
  ["psb", "application/vnd.3gpp.pic-bw-small"],
  ["psd", "application/x-photoshop"],
  ["psf", "application/x-font-linux-psf"],
  ["pskcxml", "application/pskc+xml"],
  ["pti", "image/prs.pti"],
  ["ptid", "application/vnd.pvi.ptid1"],
  ["pub", "application/x-mspublisher"],
  ["pvb", "application/vnd.3gpp.pic-bw-var"],
  ["pwn", "application/vnd.3m.post-it-notes"],
  ["pya", "audio/vnd.ms-playready.media.pya"],
  ["pyv", "video/vnd.ms-playready.media.pyv"],
  ["qam", "application/vnd.epson.quickanime"],
  ["qbo", "application/vnd.intu.qbo"],
  ["qfx", "application/vnd.intu.qfx"],
  ["qps", "application/vnd.publishare-delta-tree"],
  ["qt", "video/quicktime"],
  ["qwd", "application/vnd.quark.quarkxpress"],
  ["qwt", "application/vnd.quark.quarkxpress"],
  ["qxb", "application/vnd.quark.quarkxpress"],
  ["qxd", "application/vnd.quark.quarkxpress"],
  ["qxl", "application/vnd.quark.quarkxpress"],
  ["qxt", "application/vnd.quark.quarkxpress"],
  ["ra", "audio/x-realaudio"],
  ["ram", "audio/x-pn-realaudio"],
  ["raml", "application/raml+yaml"],
  ["rapd", "application/route-apd+xml"],
  ["rar", "application/x-rar"],
  ["ras", "image/x-cmu-raster"],
  ["rcprofile", "application/vnd.ipunplugged.rcprofile"],
  ["rdf", "application/rdf+xml"],
  ["rdz", "application/vnd.data-vision.rdz"],
  ["relo", "application/p2p-overlay+xml"],
  ["rep", "application/vnd.businessobjects"],
  ["res", "application/x-dtbresource+xml"],
  ["rgb", "image/x-rgb"],
  ["rif", "application/reginfo+xml"],
  ["rip", "audio/vnd.rip"],
  ["ris", "application/x-research-info-systems"],
  ["rl", "application/resource-lists+xml"],
  ["rlc", "image/vnd.fujixerox.edmics-rlc"],
  ["rld", "application/resource-lists-diff+xml"],
  ["rm", "audio/x-pn-realaudio"],
  ["rmi", "audio/midi"],
  ["rmp", "audio/x-pn-realaudio-plugin"],
  ["rms", "application/vnd.jcp.javame.midlet-rms"],
  ["rmvb", "application/vnd.rn-realmedia-vbr"],
  ["rnc", "application/relax-ng-compact-syntax"],
  ["rng", "application/xml"],
  ["roa", "application/rpki-roa"],
  ["roff", "text/troff"],
  ["rp9", "application/vnd.cloanto.rp9"],
  ["rpm", "audio/x-pn-realaudio-plugin"],
  ["rpss", "application/vnd.nokia.radio-presets"],
  ["rpst", "application/vnd.nokia.radio-preset"],
  ["rq", "application/sparql-query"],
  ["rs", "application/rls-services+xml"],
  ["rsa", "application/x-pkcs7"],
  ["rsat", "application/atsc-rsat+xml"],
  ["rsd", "application/rsd+xml"],
  ["rsheet", "application/urc-ressheet+xml"],
  ["rss", "application/rss+xml"],
  ["rtf", "text/rtf"],
  ["rtx", "text/richtext"],
  ["run", "application/x-makeself"],
  ["rusd", "application/route-usd+xml"],
  ["rv", "video/vnd.rn-realvideo"],
  ["s", "text/x-asm"],
  ["s3m", "audio/s3m"],
  ["saf", "application/vnd.yamaha.smaf-audio"],
  ["sass", "text/x-sass"],
  ["sbml", "application/sbml+xml"],
  ["sc", "application/vnd.ibm.secure-container"],
  ["scd", "application/x-msschedule"],
  ["scm", "application/vnd.lotus-screencam"],
  ["scq", "application/scvp-cv-request"],
  ["scs", "application/scvp-cv-response"],
  ["scss", "text/x-scss"],
  ["scurl", "text/vnd.curl.scurl"],
  ["sda", "application/vnd.stardivision.draw"],
  ["sdc", "application/vnd.stardivision.calc"],
  ["sdd", "application/vnd.stardivision.impress"],
  ["sdkd", "application/vnd.solent.sdkm+xml"],
  ["sdkm", "application/vnd.solent.sdkm+xml"],
  ["sdp", "application/sdp"],
  ["sdw", "application/vnd.stardivision.writer"],
  ["sea", "application/octet-stream"],
  ["see", "application/vnd.seemail"],
  ["seed", "application/vnd.fdsn.seed"],
  ["sema", "application/vnd.sema"],
  ["semd", "application/vnd.semd"],
  ["semf", "application/vnd.semf"],
  ["senmlx", "application/senml+xml"],
  ["sensmlx", "application/sensml+xml"],
  ["ser", "application/java-serialized-object"],
  ["setpay", "application/set-payment-initiation"],
  ["setreg", "application/set-registration-initiation"],
  ["sfd-hdstx", "application/vnd.hydrostatix.sof-data"],
  ["sfs", "application/vnd.spotfire.sfs"],
  ["sfv", "text/x-sfv"],
  ["sgi", "image/sgi"],
  ["sgl", "application/vnd.stardivision.writer-global"],
  ["sgm", "text/sgml"],
  ["sgml", "text/sgml"],
  ["sh", "application/x-sh"],
  ["shar", "application/x-shar"],
  ["shex", "text/shex"],
  ["shf", "application/shf+xml"],
  ["shtml", "text/html"],
  ["sid", "image/x-mrsid-image"],
  ["sieve", "application/sieve"],
  ["sig", "application/pgp-signature"],
  ["sil", "audio/silk"],
  ["silo", "model/mesh"],
  ["sis", "application/vnd.symbian.install"],
  ["sisx", "application/vnd.symbian.install"],
  ["sit", "application/x-stuffit"],
  ["sitx", "application/x-stuffitx"],
  ["siv", "application/sieve"],
  ["skd", "application/vnd.koan"],
  ["skm", "application/vnd.koan"],
  ["skp", "application/vnd.koan"],
  ["skt", "application/vnd.koan"],
  ["sldm", "application/vnd.ms-powerpoint.slide.macroenabled.12"],
  ["sldx", "application/vnd.openxmlformats-officedocument.presentationml.slide"],
  ["slim", "text/slim"],
  ["slm", "text/slim"],
  ["sls", "application/route-s-tsid+xml"],
  ["slt", "application/vnd.epson.salt"],
  ["sm", "application/vnd.stepmania.stepchart"],
  ["smf", "application/vnd.stardivision.math"],
  ["smi", "application/smil"],
  ["smil", "application/smil"],
  ["smv", "video/x-smv"],
  ["smzip", "application/vnd.stepmania.package"],
  ["snd", "audio/basic"],
  ["snf", "application/x-font-snf"],
  ["so", "application/octet-stream"],
  ["spc", "application/x-pkcs7-certificates"],
  ["spdx", "text/spdx"],
  ["spf", "application/vnd.yamaha.smaf-phrase"],
  ["spl", "application/x-futuresplash"],
  ["spot", "text/vnd.in3d.spot"],
  ["spp", "application/scvp-vp-response"],
  ["spq", "application/scvp-vp-request"],
  ["spx", "audio/ogg"],
  ["sql", "application/x-sql"],
  ["src", "application/x-wais-source"],
  ["srt", "application/x-subrip"],
  ["sru", "application/sru+xml"],
  ["srx", "application/sparql-results+xml"],
  ["ssdl", "application/ssdl+xml"],
  ["sse", "application/vnd.kodak-descriptor"],
  ["ssf", "application/vnd.epson.ssf"],
  ["ssml", "application/ssml+xml"],
  ["sst", "application/octet-stream"],
  ["st", "application/vnd.sailingtracker.track"],
  ["stc", "application/vnd.sun.xml.calc.template"],
  ["std", "application/vnd.sun.xml.draw.template"],
  ["stf", "application/vnd.wt.stf"],
  ["sti", "application/vnd.sun.xml.impress.template"],
  ["stk", "application/hyperstudio"],
  ["stl", "model/stl"],
  ["stpx", "model/step+xml"],
  ["stpxz", "model/step-xml+zip"],
  ["stpz", "model/step+zip"],
  ["str", "application/vnd.pg.format"],
  ["stw", "application/vnd.sun.xml.writer.template"],
  ["styl", "text/stylus"],
  ["stylus", "text/stylus"],
  ["sub", "text/vnd.dvb.subtitle"],
  ["sus", "application/vnd.sus-calendar"],
  ["susp", "application/vnd.sus-calendar"],
  ["sv4cpio", "application/x-sv4cpio"],
  ["sv4crc", "application/x-sv4crc"],
  ["svc", "application/vnd.dvb.service"],
  ["svd", "application/vnd.svd"],
  ["svg", "image/svg+xml"],
  ["svgz", "image/svg+xml"],
  ["swa", "application/x-director"],
  ["swf", "application/x-shockwave-flash"],
  ["swi", "application/vnd.aristanetworks.swi"],
  ["swidtag", "application/swid+xml"],
  ["sxc", "application/vnd.sun.xml.calc"],
  ["sxd", "application/vnd.sun.xml.draw"],
  ["sxg", "application/vnd.sun.xml.writer.global"],
  ["sxi", "application/vnd.sun.xml.impress"],
  ["sxm", "application/vnd.sun.xml.math"],
  ["sxw", "application/vnd.sun.xml.writer"],
  ["t", "text/troff"],
  ["t3", "application/x-t3vm-image"],
  ["t38", "image/t38"],
  ["taglet", "application/vnd.mynfc"],
  ["tao", "application/vnd.tao.intent-module-archive"],
  ["tap", "image/vnd.tencent.tap"],
  ["tar", "application/x-tar"],
  ["tcap", "application/vnd.3gpp2.tcap"],
  ["tcl", "application/x-tcl"],
  ["td", "application/urc-targetdesc+xml"],
  ["teacher", "application/vnd.smart.teacher"],
  ["tei", "application/tei+xml"],
  ["teicorpus", "application/tei+xml"],
  ["tex", "application/x-tex"],
  ["texi", "application/x-texinfo"],
  ["texinfo", "application/x-texinfo"],
  ["text", "text/plain"],
  ["tfi", "application/thraud+xml"],
  ["tfm", "application/x-tex-tfm"],
  ["tfx", "image/tiff-fx"],
  ["tga", "image/x-tga"],
  ["tgz", "application/x-tar"],
  ["thmx", "application/vnd.ms-officetheme"],
  ["tif", "image/tiff"],
  ["tiff", "image/tiff"],
  ["tk", "application/x-tcl"],
  ["tmo", "application/vnd.tmobile-livetv"],
  ["toml", "application/toml"],
  ["torrent", "application/x-bittorrent"],
  ["tpl", "application/vnd.groove-tool-template"],
  ["tpt", "application/vnd.trid.tpt"],
  ["tr", "text/troff"],
  ["tra", "application/vnd.trueapp"],
  ["trig", "application/trig"],
  ["trm", "application/x-msterminal"],
  ["ts", "video/mp2t"],
  ["tsd", "application/timestamped-data"],
  ["tsv", "text/tab-separated-values"],
  ["ttc", "font/collection"],
  ["ttf", "font/ttf"],
  ["ttl", "text/turtle"],
  ["ttml", "application/ttml+xml"],
  ["twd", "application/vnd.simtech-mindmapper"],
  ["twds", "application/vnd.simtech-mindmapper"],
  ["txd", "application/vnd.genomatix.tuxedo"],
  ["txf", "application/vnd.mobius.txf"],
  ["txt", "text/plain"],
  ["u8dsn", "message/global-delivery-status"],
  ["u8hdr", "message/global-headers"],
  ["u8mdn", "message/global-disposition-notification"],
  ["u8msg", "message/global"],
  ["u32", "application/x-authorware-bin"],
  ["ubj", "application/ubjson"],
  ["udeb", "application/x-debian-package"],
  ["ufd", "application/vnd.ufdl"],
  ["ufdl", "application/vnd.ufdl"],
  ["ulx", "application/x-glulx"],
  ["umj", "application/vnd.umajin"],
  ["unityweb", "application/vnd.unity"],
  ["uoml", "application/vnd.uoml+xml"],
  ["uri", "text/uri-list"],
  ["uris", "text/uri-list"],
  ["urls", "text/uri-list"],
  ["usdz", "model/vnd.usdz+zip"],
  ["ustar", "application/x-ustar"],
  ["utz", "application/vnd.uiq.theme"],
  ["uu", "text/x-uuencode"],
  ["uva", "audio/vnd.dece.audio"],
  ["uvd", "application/vnd.dece.data"],
  ["uvf", "application/vnd.dece.data"],
  ["uvg", "image/vnd.dece.graphic"],
  ["uvh", "video/vnd.dece.hd"],
  ["uvi", "image/vnd.dece.graphic"],
  ["uvm", "video/vnd.dece.mobile"],
  ["uvp", "video/vnd.dece.pd"],
  ["uvs", "video/vnd.dece.sd"],
  ["uvt", "application/vnd.dece.ttml+xml"],
  ["uvu", "video/vnd.uvvu.mp4"],
  ["uvv", "video/vnd.dece.video"],
  ["uvva", "audio/vnd.dece.audio"],
  ["uvvd", "application/vnd.dece.data"],
  ["uvvf", "application/vnd.dece.data"],
  ["uvvg", "image/vnd.dece.graphic"],
  ["uvvh", "video/vnd.dece.hd"],
  ["uvvi", "image/vnd.dece.graphic"],
  ["uvvm", "video/vnd.dece.mobile"],
  ["uvvp", "video/vnd.dece.pd"],
  ["uvvs", "video/vnd.dece.sd"],
  ["uvvt", "application/vnd.dece.ttml+xml"],
  ["uvvu", "video/vnd.uvvu.mp4"],
  ["uvvv", "video/vnd.dece.video"],
  ["uvvx", "application/vnd.dece.unspecified"],
  ["uvvz", "application/vnd.dece.zip"],
  ["uvx", "application/vnd.dece.unspecified"],
  ["uvz", "application/vnd.dece.zip"],
  ["vbox", "application/x-virtualbox-vbox"],
  ["vbox-extpack", "application/x-virtualbox-vbox-extpack"],
  ["vcard", "text/vcard"],
  ["vcd", "application/x-cdlink"],
  ["vcf", "text/x-vcard"],
  ["vcg", "application/vnd.groove-vcard"],
  ["vcs", "text/x-vcalendar"],
  ["vcx", "application/vnd.vcx"],
  ["vdi", "application/x-virtualbox-vdi"],
  ["vds", "model/vnd.sap.vds"],
  ["vhd", "application/x-virtualbox-vhd"],
  ["vis", "application/vnd.visionary"],
  ["viv", "video/vnd.vivo"],
  ["vlc", "application/videolan"],
  ["vmdk", "application/x-virtualbox-vmdk"],
  ["vob", "video/x-ms-vob"],
  ["vor", "application/vnd.stardivision.writer"],
  ["vox", "application/x-authorware-bin"],
  ["vrml", "model/vrml"],
  ["vsd", "application/vnd.visio"],
  ["vsf", "application/vnd.vsf"],
  ["vss", "application/vnd.visio"],
  ["vst", "application/vnd.visio"],
  ["vsw", "application/vnd.visio"],
  ["vtf", "image/vnd.valve.source.texture"],
  ["vtt", "text/vtt"],
  ["vtu", "model/vnd.vtu"],
  ["vxml", "application/voicexml+xml"],
  ["w3d", "application/x-director"],
  ["wad", "application/x-doom"],
  ["wadl", "application/vnd.sun.wadl+xml"],
  ["war", "application/java-archive"],
  ["wasm", "application/wasm"],
  ["wav", "audio/x-wav"],
  ["wax", "audio/x-ms-wax"],
  ["wbmp", "image/vnd.wap.wbmp"],
  ["wbs", "application/vnd.criticaltools.wbs+xml"],
  ["wbxml", "application/wbxml"],
  ["wcm", "application/vnd.ms-works"],
  ["wdb", "application/vnd.ms-works"],
  ["wdp", "image/vnd.ms-photo"],
  ["weba", "audio/webm"],
  ["webapp", "application/x-web-app-manifest+json"],
  ["webm", "video/webm"],
  ["webmanifest", "application/manifest+json"],
  ["webp", "image/webp"],
  ["wg", "application/vnd.pmi.widget"],
  ["wgt", "application/widget"],
  ["wks", "application/vnd.ms-works"],
  ["wm", "video/x-ms-wm"],
  ["wma", "audio/x-ms-wma"],
  ["wmd", "application/x-ms-wmd"],
  ["wmf", "image/wmf"],
  ["wml", "text/vnd.wap.wml"],
  ["wmlc", "application/wmlc"],
  ["wmls", "text/vnd.wap.wmlscript"],
  ["wmlsc", "application/vnd.wap.wmlscriptc"],
  ["wmv", "video/x-ms-wmv"],
  ["wmx", "video/x-ms-wmx"],
  ["wmz", "application/x-msmetafile"],
  ["woff", "font/woff"],
  ["woff2", "font/woff2"],
  ["word", "application/msword"],
  ["wpd", "application/vnd.wordperfect"],
  ["wpl", "application/vnd.ms-wpl"],
  ["wps", "application/vnd.ms-works"],
  ["wqd", "application/vnd.wqd"],
  ["wri", "application/x-mswrite"],
  ["wrl", "model/vrml"],
  ["wsc", "message/vnd.wfa.wsc"],
  ["wsdl", "application/wsdl+xml"],
  ["wspolicy", "application/wspolicy+xml"],
  ["wtb", "application/vnd.webturbo"],
  ["wvx", "video/x-ms-wvx"],
  ["x3d", "model/x3d+xml"],
  ["x3db", "model/x3d+fastinfoset"],
  ["x3dbz", "model/x3d+binary"],
  ["x3dv", "model/x3d-vrml"],
  ["x3dvz", "model/x3d+vrml"],
  ["x3dz", "model/x3d+xml"],
  ["x32", "application/x-authorware-bin"],
  ["x_b", "model/vnd.parasolid.transmit.binary"],
  ["x_t", "model/vnd.parasolid.transmit.text"],
  ["xaml", "application/xaml+xml"],
  ["xap", "application/x-silverlight-app"],
  ["xar", "application/vnd.xara"],
  ["xav", "application/xcap-att+xml"],
  ["xbap", "application/x-ms-xbap"],
  ["xbd", "application/vnd.fujixerox.docuworks.binder"],
  ["xbm", "image/x-xbitmap"],
  ["xca", "application/xcap-caps+xml"],
  ["xcs", "application/calendar+xml"],
  ["xdf", "application/xcap-diff+xml"],
  ["xdm", "application/vnd.syncml.dm+xml"],
  ["xdp", "application/vnd.adobe.xdp+xml"],
  ["xdssc", "application/dssc+xml"],
  ["xdw", "application/vnd.fujixerox.docuworks"],
  ["xel", "application/xcap-el+xml"],
  ["xenc", "application/xenc+xml"],
  ["xer", "application/patch-ops-error+xml"],
  ["xfdf", "application/vnd.adobe.xfdf"],
  ["xfdl", "application/vnd.xfdl"],
  ["xht", "application/xhtml+xml"],
  ["xhtml", "application/xhtml+xml"],
  ["xhvml", "application/xv+xml"],
  ["xif", "image/vnd.xiff"],
  ["xl", "application/excel"],
  ["xla", "application/vnd.ms-excel"],
  ["xlam", "application/vnd.ms-excel.addin.macroEnabled.12"],
  ["xlc", "application/vnd.ms-excel"],
  ["xlf", "application/xliff+xml"],
  ["xlm", "application/vnd.ms-excel"],
  ["xls", "application/vnd.ms-excel"],
  ["xlsb", "application/vnd.ms-excel.sheet.binary.macroEnabled.12"],
  ["xlsm", "application/vnd.ms-excel.sheet.macroEnabled.12"],
  ["xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
  ["xlt", "application/vnd.ms-excel"],
  ["xltm", "application/vnd.ms-excel.template.macroEnabled.12"],
  ["xltx", "application/vnd.openxmlformats-officedocument.spreadsheetml.template"],
  ["xlw", "application/vnd.ms-excel"],
  ["xm", "audio/xm"],
  ["xml", "application/xml"],
  ["xns", "application/xcap-ns+xml"],
  ["xo", "application/vnd.olpc-sugar"],
  ["xop", "application/xop+xml"],
  ["xpi", "application/x-xpinstall"],
  ["xpl", "application/xproc+xml"],
  ["xpm", "image/x-xpixmap"],
  ["xpr", "application/vnd.is-xpr"],
  ["xps", "application/vnd.ms-xpsdocument"],
  ["xpw", "application/vnd.intercon.formnet"],
  ["xpx", "application/vnd.intercon.formnet"],
  ["xsd", "application/xml"],
  ["xsl", "application/xml"],
  ["xslt", "application/xslt+xml"],
  ["xsm", "application/vnd.syncml+xml"],
  ["xspf", "application/xspf+xml"],
  ["xul", "application/vnd.mozilla.xul+xml"],
  ["xvm", "application/xv+xml"],
  ["xvml", "application/xv+xml"],
  ["xwd", "image/x-xwindowdump"],
  ["xyz", "chemical/x-xyz"],
  ["xz", "application/x-xz"],
  ["yaml", "text/yaml"],
  ["yang", "application/yang"],
  ["yin", "application/yin+xml"],
  ["yml", "text/yaml"],
  ["ymp", "text/x-suse-ymp"],
  ["z", "application/x-compress"],
  ["z1", "application/x-zmachine"],
  ["z2", "application/x-zmachine"],
  ["z3", "application/x-zmachine"],
  ["z4", "application/x-zmachine"],
  ["z5", "application/x-zmachine"],
  ["z6", "application/x-zmachine"],
  ["z7", "application/x-zmachine"],
  ["z8", "application/x-zmachine"],
  ["zaz", "application/vnd.zzazz.deck+xml"],
  ["zip", "application/zip"],
  ["zir", "application/vnd.zul"],
  ["zirz", "application/vnd.zul"],
  ["zmm", "application/vnd.handheld-entertainment+xml"],
  ["zsh", "text/x-scriptzsh"]
]);
function hr(t, e, n) {
  const o = Ty(t), { webkitRelativePath: d } = t, g = typeof e == "string" ? e : typeof d == "string" && d.length > 0 ? d : `./${t.name}`;
  return typeof o.path != "string" && Xh(o, "path", g), Xh(o, "relativePath", g), o;
}
function Ty(t) {
  const { name: e } = t;
  if (e && e.lastIndexOf(".") !== -1 && !t.type) {
    const n = e.split(".").pop().toLowerCase(), o = ky.get(n);
    o && Object.defineProperty(t, "type", {
      value: o,
      writable: !1,
      configurable: !1,
      enumerable: !0
    });
  }
  return t;
}
function Xh(t, e, n) {
  Object.defineProperty(t, e, {
    value: n,
    writable: !1,
    configurable: !1,
    enumerable: !0
  });
}
const Oy = [
  // Thumbnail cache files for macOS and Windows
  ".DS_Store",
  // macOs
  "Thumbs.db"
  // Windows
];
function Py(t) {
  return ji(this, void 0, void 0, function* () {
    return Fo(t) && My(t.dataTransfer) ? Ly(t.dataTransfer, t.type) : Ay(t) ? Dy(t) : Array.isArray(t) && t.every((e) => "getFile" in e && typeof e.getFile == "function") ? jy(t) : [];
  });
}
function My(t) {
  return Fo(t);
}
function Ay(t) {
  return Fo(t) && Fo(t.target);
}
function Fo(t) {
  return typeof t == "object" && t !== null;
}
function Dy(t) {
  return Ul(t.target.files).map((e) => hr(e));
}
function jy(t) {
  return ji(this, void 0, void 0, function* () {
    return (yield Promise.all(t.map((e) => e.getFile()))).map((e) => hr(e));
  });
}
function Ly(t, e) {
  return ji(this, void 0, void 0, function* () {
    if (t.items) {
      const n = Ul(t.items).filter((d) => d.kind === "file");
      if (e !== "drop")
        return n;
      const o = yield Promise.all(n.map(Fy));
      return Yh($p(o));
    }
    return Yh(Ul(t.files).map((n) => hr(n)));
  });
}
function Yh(t) {
  return t.filter((e) => Oy.indexOf(e.name) === -1);
}
function Ul(t) {
  if (t === null)
    return [];
  const e = [];
  for (let n = 0; n < t.length; n++) {
    const o = t[n];
    e.push(o);
  }
  return e;
}
function Fy(t) {
  if (typeof t.webkitGetAsEntry != "function")
    return Vh(t);
  const e = t.webkitGetAsEntry();
  return e && e.isDirectory ? t0(e) : Vh(t, e);
}
function $p(t) {
  return t.reduce((e, n) => [
    ...e,
    ...Array.isArray(n) ? $p(n) : [n]
  ], []);
}
function Vh(t, e) {
  var n;
  if (typeof t.getAsFileSystemHandle == "function")
    return t.getAsFileSystemHandle().then((g) => ji(this, void 0, void 0, function* () {
      const b = yield g.getFile();
      return b.handle = g, hr(b);
    }));
  const o = t.getAsFile();
  if (!o)
    return Promise.reject(`${t} is not a File`);
  const d = hr(o, (n = e == null ? void 0 : e.fullPath) !== null && n !== void 0 ? n : void 0);
  return Promise.resolve(d);
}
function Iy(t) {
  return ji(this, void 0, void 0, function* () {
    return t.isDirectory ? t0(t) : Ry(t);
  });
}
function t0(t) {
  const e = t.createReader();
  return new Promise((n, o) => {
    const d = [];
    function g() {
      e.readEntries((b) => ji(this, void 0, void 0, function* () {
        if (b.length) {
          const _ = Promise.all(b.map(Iy));
          d.push(_), g();
        } else
          try {
            const _ = yield Promise.all(d);
            n(_);
          } catch (_) {
            o(_);
          }
      }), (b) => {
        o(b);
      });
    }
    g();
  });
}
function Ry(t) {
  return ji(this, void 0, void 0, function* () {
    return new Promise((e, n) => {
      t.file((o) => {
        const d = hr(o, t.fullPath);
        e(d);
      }, (o) => {
        n(o);
      });
    });
  });
}
var zs = function (t, e) {
  if (t && e) {
    var n = Array.isArray(e) ? e : e.split(",");
    if (n.length === 0)
      return !0;
    var o = t.name || "", d = (t.type || "").toLowerCase(), g = d.replace(/\/.*$/, "");
    return n.some(function (b) {
      var _ = b.trim().toLowerCase();
      return _.charAt(0) === "." ? o.toLowerCase().endsWith(_) : _.endsWith("/*") ? g === _.replace(/\/.*$/, "") : d === _;
    });
  }
  return !0;
};
function Gh(t) {
  return Ny(t) || zy(t) || n0(t) || By();
}
function By() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function zy(t) {
  if (typeof Symbol < "u" && t[Symbol.iterator] != null || t["@@iterator"] != null) return Array.from(t);
}
function Ny(t) {
  if (Array.isArray(t)) return Hl(t);
}
function qh(t, e) {
  var n = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(t);
    e && (o = o.filter(function (d) {
      return Object.getOwnPropertyDescriptor(t, d).enumerable;
    })), n.push.apply(n, o);
  }
  return n;
}
function Zh(t) {
  for (var e = 1; e < arguments.length; e++) {
    var n = arguments[e] != null ? arguments[e] : {};
    e % 2 ? qh(Object(n), !0).forEach(function (o) {
      e0(t, o, n[o]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(n)) : qh(Object(n)).forEach(function (o) {
      Object.defineProperty(t, o, Object.getOwnPropertyDescriptor(n, o));
    });
  }
  return t;
}
function e0(t, e, n) {
  return e in t ? Object.defineProperty(t, e, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : t[e] = n, t;
}
function ua(t, e) {
  return Wy(t) || Hy(t, e) || n0(t, e) || Uy();
}
function Uy() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function n0(t, e) {
  if (t) {
    if (typeof t == "string") return Hl(t, e);
    var n = Object.prototype.toString.call(t).slice(8, -1);
    if (n === "Object" && t.constructor && (n = t.constructor.name), n === "Map" || n === "Set") return Array.from(t);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return Hl(t, e);
  }
}
function Hl(t, e) {
  (e == null || e > t.length) && (e = t.length);
  for (var n = 0, o = new Array(e); n < e; n++)
    o[n] = t[n];
  return o;
}
function Hy(t, e) {
  var n = t == null ? null : typeof Symbol < "u" && t[Symbol.iterator] || t["@@iterator"];
  if (n != null) {
    var o = [], d = !0, g = !1, b, _;
    try {
      for (n = n.call(t); !(d = (b = n.next()).done) && (o.push(b.value), !(e && o.length === e)); d = !0)
        ;
    } catch (w) {
      g = !0, _ = w;
    } finally {
      try {
        !d && n.return != null && n.return();
      } finally {
        if (g) throw _;
      }
    }
    return o;
  }
}
function Wy(t) {
  if (Array.isArray(t)) return t;
}
var Xy = typeof zs == "function" ? zs : zs.default, Yy = "file-invalid-type", Vy = "file-too-large", Gy = "file-too-small", qy = "too-many-files", Zy = function () {
  var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", e = t.split(","), n = e.length > 1 ? "one of ".concat(e.join(", ")) : e[0];
  return {
    code: Yy,
    message: "File type must be ".concat(n)
  };
}, Qh = function (t) {
  return {
    code: Vy,
    message: "File is larger than ".concat(t, " ").concat(t === 1 ? "byte" : "bytes")
  };
}, Kh = function (t) {
  return {
    code: Gy,
    message: "File is smaller than ".concat(t, " ").concat(t === 1 ? "byte" : "bytes")
  };
}, Qy = {
  code: qy,
  message: "Too many files"
};
function i0(t, e) {
  var n = t.type === "application/x-moz-file" || Xy(t, e);
  return [n, n ? null : Zy(e)];
}
function r0(t, e, n) {
  if (bi(t.size))
    if (bi(e) && bi(n)) {
      if (t.size > n) return [!1, Qh(n)];
      if (t.size < e) return [!1, Kh(e)];
    } else {
      if (bi(e) && t.size < e) return [!1, Kh(e)];
      if (bi(n) && t.size > n) return [!1, Qh(n)];
    }
  return [!0, null];
}
function bi(t) {
  return t != null;
}
function Ky(t) {
  var e = t.files, n = t.accept, o = t.minSize, d = t.maxSize, g = t.multiple, b = t.maxFiles, _ = t.validator;
  return !g && e.length > 1 || g && b >= 1 && e.length > b ? !1 : e.every(function (w) {
    var E = i0(w, n), j = ua(E, 1), H = j[0], G = r0(w, o, d), J = ua(G, 1), nt = J[0], rt = _ ? _(w) : null;
    return H && nt && !rt;
  });
}
function Io(t) {
  return typeof t.isPropagationStopped == "function" ? t.isPropagationStopped() : typeof t.cancelBubble < "u" ? t.cancelBubble : !1;
}
function Ya(t) {
  return t.dataTransfer ? Array.prototype.some.call(t.dataTransfer.types, function (e) {
    return e === "Files" || e === "application/x-moz-file";
  }) : !!t.target && !!t.target.files;
}
function Jh(t) {
  t.preventDefault();
}
function Jy(t) {
  return t.indexOf("MSIE") !== -1 || t.indexOf("Trident/") !== -1;
}
function $y(t) {
  return t.indexOf("Edge/") !== -1;
}
function tb() {
  var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : window.navigator.userAgent;
  return Jy(t) || $y(t);
}
function Cn() {
  for (var t = arguments.length, e = new Array(t), n = 0; n < t; n++)
    e[n] = arguments[n];
  return function (o) {
    for (var d = arguments.length, g = new Array(d > 1 ? d - 1 : 0), b = 1; b < d; b++)
      g[b - 1] = arguments[b];
    return e.some(function (_) {
      return !Io(o) && _ && _.apply(void 0, [o].concat(g)), Io(o);
    });
  };
}
function eb() {
  return "showOpenFilePicker" in window;
}
function nb(t) {
  if (bi(t)) {
    var e = Object.entries(t).filter(function (n) {
      var o = ua(n, 2), d = o[0], g = o[1], b = !0;
      return a0(d) || (console.warn('Skipped "'.concat(d, '" because it is not a valid MIME type. Check https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types for a list of valid MIME types.')), b = !1), (!Array.isArray(g) || !g.every(o0)) && (console.warn('Skipped "'.concat(d, '" because an invalid file extension was provided.')), b = !1), b;
    }).reduce(function (n, o) {
      var d = ua(o, 2), g = d[0], b = d[1];
      return Zh(Zh({}, n), {}, e0({}, g, b));
    }, {});
    return [{
      // description is required due to https://crbug.com/1264708
      description: "Files",
      accept: e
    }];
  }
  return t;
}
function ib(t) {
  if (bi(t))
    return Object.entries(t).reduce(function (e, n) {
      var o = ua(n, 2), d = o[0], g = o[1];
      return [].concat(Gh(e), [d], Gh(g));
    }, []).filter(function (e) {
      return a0(e) || o0(e);
    }).join(",");
}
function rb(t) {
  return t instanceof DOMException && (t.name === "AbortError" || t.code === t.ABORT_ERR);
}
function ab(t) {
  return t instanceof DOMException && (t.name === "SecurityError" || t.code === t.SECURITY_ERR);
}
function a0(t) {
  return t === "audio/*" || t === "video/*" || t === "image/*" || t === "text/*" || t === "application/*" || /\w+\/[-+.\w]+/g.test(t);
}
function o0(t) {
  return /^.*\.[\w]+$/.test(t);
}
var ob = ["children"], sb = ["open"], lb = ["refKey", "role", "onKeyDown", "onFocus", "onBlur", "onClick", "onDragEnter", "onDragOver", "onDragLeave", "onDrop"], cb = ["refKey", "onChange", "onClick"];
function ub(t) {
  return db(t) || fb(t) || s0(t) || hb();
}
function hb() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function fb(t) {
  if (typeof Symbol < "u" && t[Symbol.iterator] != null || t["@@iterator"] != null) return Array.from(t);
}
function db(t) {
  if (Array.isArray(t)) return Wl(t);
}
function Ns(t, e) {
  return mb(t) || gb(t, e) || s0(t, e) || pb();
}
function pb() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function s0(t, e) {
  if (t) {
    if (typeof t == "string") return Wl(t, e);
    var n = Object.prototype.toString.call(t).slice(8, -1);
    if (n === "Object" && t.constructor && (n = t.constructor.name), n === "Map" || n === "Set") return Array.from(t);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return Wl(t, e);
  }
}
function Wl(t, e) {
  (e == null || e > t.length) && (e = t.length);
  for (var n = 0, o = new Array(e); n < e; n++)
    o[n] = t[n];
  return o;
}
function gb(t, e) {
  var n = t == null ? null : typeof Symbol < "u" && t[Symbol.iterator] || t["@@iterator"];
  if (n != null) {
    var o = [], d = !0, g = !1, b, _;
    try {
      for (n = n.call(t); !(d = (b = n.next()).done) && (o.push(b.value), !(e && o.length === e)); d = !0)
        ;
    } catch (w) {
      g = !0, _ = w;
    } finally {
      try {
        !d && n.return != null && n.return();
      } finally {
        if (g) throw _;
      }
    }
    return o;
  }
}
function mb(t) {
  if (Array.isArray(t)) return t;
}
function $h(t, e) {
  var n = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(t);
    e && (o = o.filter(function (d) {
      return Object.getOwnPropertyDescriptor(t, d).enumerable;
    })), n.push.apply(n, o);
  }
  return n;
}
function Jt(t) {
  for (var e = 1; e < arguments.length; e++) {
    var n = arguments[e] != null ? arguments[e] : {};
    e % 2 ? $h(Object(n), !0).forEach(function (o) {
      Xl(t, o, n[o]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(n)) : $h(Object(n)).forEach(function (o) {
      Object.defineProperty(t, o, Object.getOwnPropertyDescriptor(n, o));
    });
  }
  return t;
}
function Xl(t, e, n) {
  return e in t ? Object.defineProperty(t, e, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : t[e] = n, t;
}
function Ro(t, e) {
  if (t == null) return {};
  var n = vb(t, e), o, d;
  if (Object.getOwnPropertySymbols) {
    var g = Object.getOwnPropertySymbols(t);
    for (d = 0; d < g.length; d++)
      o = g[d], !(e.indexOf(o) >= 0) && Object.prototype.propertyIsEnumerable.call(t, o) && (n[o] = t[o]);
  }
  return n;
}
function vb(t, e) {
  if (t == null) return {};
  var n = {}, o = Object.keys(t), d, g;
  for (g = 0; g < o.length; g++)
    d = o[g], !(e.indexOf(d) >= 0) && (n[d] = t[d]);
  return n;
}
var Wc = /* @__PURE__ */ P.forwardRef(function (t, e) {
  var n = t.children, o = Ro(t, ob), d = c0(o), g = d.open, b = Ro(d, sb);
  return P.useImperativeHandle(e, function () {
    return {
      open: g
    };
  }, [g]), /* @__PURE__ */ Ff.createElement(P.Fragment, null, n(Jt(Jt({}, b), {}, {
    open: g
  })));
});
Wc.displayName = "Dropzone";
var l0 = {
  disabled: !1,
  getFilesFromEvent: Py,
  maxSize: 1 / 0,
  minSize: 0,
  multiple: !0,
  maxFiles: 0,
  preventDropOnDocument: !0,
  noClick: !1,
  noKeyboard: !1,
  noDrag: !1,
  noDragEventsBubbling: !1,
  validator: null,
  useFsAccessApi: !1,
  autoFocus: !1
};
Wc.defaultProps = l0;
Wc.propTypes = {
  /**
   * Render function that exposes the dropzone state and prop getter fns
   *
   * @param {object} params
   * @param {Function} params.getRootProps Returns the props you should apply to the root drop container you render
   * @param {Function} params.getInputProps Returns the props you should apply to hidden file input you render
   * @param {Function} params.open Open the native file selection dialog
   * @param {boolean} params.isFocused Dropzone area is in focus
   * @param {boolean} params.isFileDialogActive File dialog is opened
   * @param {boolean} params.isDragActive Active drag is in progress
   * @param {boolean} params.isDragAccept Dragged files are accepted
   * @param {boolean} params.isDragReject Some dragged files are rejected
   * @param {File[]} params.acceptedFiles Accepted files
   * @param {FileRejection[]} params.fileRejections Rejected files and why they were rejected
   */
  children: Yt.func,
  /**
   * Set accepted file types.
   * Checkout https://developer.mozilla.org/en-US/docs/Web/API/window/showOpenFilePicker types option for more information.
   * Keep in mind that mime type determination is not reliable across platforms. CSV files,
   * for example, are reported as text/plain under macOS but as application/vnd.ms-excel under
   * Windows. In some cases there might not be a mime type set at all (https://github.com/react-dropzone/react-dropzone/issues/276).
   */
  accept: Yt.objectOf(Yt.arrayOf(Yt.string)),
  /**
   * Allow drag 'n' drop (or selection from the file dialog) of multiple files
   */
  multiple: Yt.bool,
  /**
   * If false, allow dropped items to take over the current browser window
   */
  preventDropOnDocument: Yt.bool,
  /**
   * If true, disables click to open the native file selection dialog
   */
  noClick: Yt.bool,
  /**
   * If true, disables SPACE/ENTER to open the native file selection dialog.
   * Note that it also stops tracking the focus state.
   */
  noKeyboard: Yt.bool,
  /**
   * If true, disables drag 'n' drop
   */
  noDrag: Yt.bool,
  /**
   * If true, stops drag event propagation to parents
   */
  noDragEventsBubbling: Yt.bool,
  /**
   * Minimum file size (in bytes)
   */
  minSize: Yt.number,
  /**
   * Maximum file size (in bytes)
   */
  maxSize: Yt.number,
  /**
   * Maximum accepted number of files
   * The default value is 0 which means there is no limitation to how many files are accepted.
   */
  maxFiles: Yt.number,
  /**
   * Enable/disable the dropzone
   */
  disabled: Yt.bool,
  /**
   * Use this to provide a custom file aggregator
   *
   * @param {(DragEvent|Event)} event A drag event or input change event (if files were selected via the file dialog)
   */
  getFilesFromEvent: Yt.func,
  /**
   * Cb for when closing the file dialog with no selection
   */
  onFileDialogCancel: Yt.func,
  /**
   * Cb for when opening the file dialog
   */
  onFileDialogOpen: Yt.func,
  /**
   * Set to true to use the https://developer.mozilla.org/en-US/docs/Web/API/File_System_Access_API
   * to open the file picker instead of using an `<input type="file">` click event.
   */
  useFsAccessApi: Yt.bool,
  /**
   * Set to true to focus the root element on render
   */
  autoFocus: Yt.bool,
  /**
   * Cb for when the `dragenter` event occurs.
   *
   * @param {DragEvent} event
   */
  onDragEnter: Yt.func,
  /**
   * Cb for when the `dragleave` event occurs
   *
   * @param {DragEvent} event
   */
  onDragLeave: Yt.func,
  /**
   * Cb for when the `dragover` event occurs
   *
   * @param {DragEvent} event
   */
  onDragOver: Yt.func,
  /**
   * Cb for when the `drop` event occurs.
   * Note that this callback is invoked after the `getFilesFromEvent` callback is done.
   *
   * Files are accepted or rejected based on the `accept`, `multiple`, `minSize` and `maxSize` props.
   * `accept` must be a valid [MIME type](http://www.iana.org/assignments/media-types/media-types.xhtml) according to [input element specification](https://www.w3.org/wiki/HTML/Elements/input/file) or a valid file extension.
   * If `multiple` is set to false and additional files are dropped,
   * all files besides the first will be rejected.
   * Any file which does not have a size in the [`minSize`, `maxSize`] range, will be rejected as well.
   *
   * Note that the `onDrop` callback will always be invoked regardless if the dropped files were accepted or rejected.
   * If you'd like to react to a specific scenario, use the `onDropAccepted`/`onDropRejected` props.
   *
   * `onDrop` will provide you with an array of [File](https://developer.mozilla.org/en-US/docs/Web/API/File) objects which you can then process and send to a server.
   * For example, with [SuperAgent](https://github.com/visionmedia/superagent) as a http/ajax library:
   *
   * ```js
   * function onDrop(acceptedFiles) {
   *   const req = request.post('/upload')
   *   acceptedFiles.forEach(file => {
   *     req.attach(file.name, file)
   *   })
   *   req.end(callback)
   * }
   * ```
   *
   * @param {File[]} acceptedFiles
   * @param {FileRejection[]} fileRejections
   * @param {(DragEvent|Event)} event A drag event or input change event (if files were selected via the file dialog)
   */
  onDrop: Yt.func,
  /**
   * Cb for when the `drop` event occurs.
   * Note that if no files are accepted, this callback is not invoked.
   *
   * @param {File[]} files
   * @param {(DragEvent|Event)} event
   */
  onDropAccepted: Yt.func,
  /**
   * Cb for when the `drop` event occurs.
   * Note that if no files are rejected, this callback is not invoked.
   *
   * @param {FileRejection[]} fileRejections
   * @param {(DragEvent|Event)} event
   */
  onDropRejected: Yt.func,
  /**
   * Cb for when there's some error from any of the promises.
   *
   * @param {Error} error
   */
  onError: Yt.func,
  /**
   * Custom validation function. It must return null if there's no errors.
   * @param {File} file
   * @returns {FileError|FileError[]|null}
   */
  validator: Yt.func
};
var Yl = {
  isFocused: !1,
  isFileDialogActive: !1,
  isDragActive: !1,
  isDragAccept: !1,
  isDragReject: !1,
  acceptedFiles: [],
  fileRejections: []
};
function c0() {
  var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, e = Jt(Jt({}, l0), t), n = e.accept, o = e.disabled, d = e.getFilesFromEvent, g = e.maxSize, b = e.minSize, _ = e.multiple, w = e.maxFiles, E = e.onDragEnter, j = e.onDragLeave, H = e.onDragOver, G = e.onDrop, J = e.onDropAccepted, nt = e.onDropRejected, rt = e.onFileDialogCancel, vt = e.onFileDialogOpen, Y = e.useFsAccessApi, L = e.autoFocus, A = e.preventDropOnDocument, tt = e.noClick, dt = e.noKeyboard, it = e.noDrag, ht = e.noDragEventsBubbling, v = e.onError, Ct = e.validator, F = P.useMemo(function () {
    return ib(n);
  }, [n]), Et = P.useMemo(function () {
    return nb(n);
  }, [n]), Ht = P.useMemo(function () {
    return typeof vt == "function" ? vt : tf;
  }, [vt]), It = P.useMemo(function () {
    return typeof rt == "function" ? rt : tf;
  }, [rt]), l = P.useRef(null), a = P.useRef(null), c = P.useReducer(yb, Yl), f = Ns(c, 2), i = f[0], r = f[1], s = i.isFocused, u = i.isFileDialogActive, h = P.useRef(typeof window < "u" && window.isSecureContext && Y && eb()), p = function () {
    !h.current && u && setTimeout(function () {
      if (a.current) {
        var z = a.current.files;
        z.length || (r({
          type: "closeDialog"
        }), It());
      }
    }, 300);
  };
  P.useEffect(function () {
    return window.addEventListener("focus", p, !1), function () {
      window.removeEventListener("focus", p, !1);
    };
  }, [a, u, It, h]);
  var m = P.useRef([]), y = function (z) {
    l.current && l.current.contains(z.target) || (z.preventDefault(), m.current = []);
  };
  P.useEffect(function () {
    return A && (document.addEventListener("dragover", Jh, !1), document.addEventListener("drop", y, !1)), function () {
      A && (document.removeEventListener("dragover", Jh), document.removeEventListener("drop", y));
    };
  }, [l, A]), P.useEffect(function () {
    return !o && L && l.current && l.current.focus(), function () {
    };
  }, [l, L, o]);
  var x = P.useCallback(function (z) {
    v ? v(z) : console.error(z);
  }, [v]), S = P.useCallback(function (z) {
    z.preventDefault(), z.persist(), K(z), m.current = [].concat(ub(m.current), [z.target]), Ya(z) && Promise.resolve(d(z)).then(function (V) {
      if (!(Io(z) && !ht)) {
        var et = V.length, lt = et > 0 && Ky({
          files: V,
          accept: F,
          minSize: b,
          maxSize: g,
          multiple: _,
          maxFiles: w,
          validator: Ct
        }), at = et > 0 && !lt;
        r({
          isDragAccept: lt,
          isDragReject: at,
          isDragActive: !0,
          type: "setDraggedFiles"
        }), E && E(z);
      }
    }).catch(function (V) {
      return x(V);
    });
  }, [d, E, x, ht, F, b, g, _, w, Ct]), k = P.useCallback(function (z) {
    z.preventDefault(), z.persist(), K(z);
    var V = Ya(z);
    if (V && z.dataTransfer)
      try {
        z.dataTransfer.dropEffect = "copy";
      } catch {
      }
    return V && H && H(z), !1;
  }, [H, ht]), D = P.useCallback(function (z) {
    z.preventDefault(), z.persist(), K(z);
    var V = m.current.filter(function (lt) {
      return l.current && l.current.contains(lt);
    }), et = V.indexOf(z.target);
    et !== -1 && V.splice(et, 1), m.current = V, !(V.length > 0) && (r({
      type: "setDraggedFiles",
      isDragActive: !1,
      isDragAccept: !1,
      isDragReject: !1
    }), Ya(z) && j && j(z));
  }, [l, j, ht]), M = P.useCallback(function (z, V) {
    var et = [], lt = [];
    z.forEach(function (at) {
      var ft = i0(at, F), ot = Ns(ft, 2), ct = ot[0], st = ot[1], yt = r0(at, b, g), gt = Ns(yt, 2), Tt = gt[0], kt = gt[1], jt = Ct ? Ct(at) : null;
      if (ct && Tt && !jt)
        et.push(at);
      else {
        var Kt = [st, kt];
        jt && (Kt = Kt.concat(jt)), lt.push({
          file: at,
          errors: Kt.filter(function (Rt) {
            return Rt;
          })
        });
      }
    }), (!_ && et.length > 1 || _ && w >= 1 && et.length > w) && (et.forEach(function (at) {
      lt.push({
        file: at,
        errors: [Qy]
      });
    }), et.splice(0)), r({
      acceptedFiles: et,
      fileRejections: lt,
      isDragReject: lt.length > 0,
      type: "setFiles"
    }), G && G(et, lt, V), lt.length > 0 && nt && nt(lt, V), et.length > 0 && J && J(et, V);
  }, [r, _, F, b, g, w, G, J, nt, Ct]), C = P.useCallback(function (z) {
    z.preventDefault(), z.persist(), K(z), m.current = [], Ya(z) && Promise.resolve(d(z)).then(function (V) {
      Io(z) && !ht || M(V, z);
    }).catch(function (V) {
      return x(V);
    }), r({
      type: "reset"
    });
  }, [d, M, x, ht]), O = P.useCallback(function () {
    if (h.current) {
      r({
        type: "openDialog"
      }), Ht();
      var z = {
        multiple: _,
        types: Et
      };
      window.showOpenFilePicker(z).then(function (V) {
        return d(V);
      }).then(function (V) {
        M(V, null), r({
          type: "closeDialog"
        });
      }).catch(function (V) {
        rb(V) ? (It(V), r({
          type: "closeDialog"
        })) : ab(V) ? (h.current = !1, a.current ? (a.current.value = null, a.current.click()) : x(new Error("Cannot open the file picker because the https://developer.mozilla.org/en-US/docs/Web/API/File_System_Access_API is not supported and no <input> was provided."))) : x(V);
      });
      return;
    }
    a.current && (r({
      type: "openDialog"
    }), Ht(), a.current.value = null, a.current.click());
  }, [r, Ht, It, Y, M, x, Et, _]), N = P.useCallback(function (z) {
    !l.current || !l.current.isEqualNode(z.target) || (z.key === " " || z.key === "Enter" || z.keyCode === 32 || z.keyCode === 13) && (z.preventDefault(), O());
  }, [l, O]), W = P.useCallback(function () {
    r({
      type: "focus"
    });
  }, []), X = P.useCallback(function () {
    r({
      type: "blur"
    });
  }, []), I = P.useCallback(function () {
    tt || (tb() ? setTimeout(O, 0) : O());
  }, [tt, O]), T = function (z) {
    return o ? null : z;
  }, R = function (z) {
    return dt ? null : T(z);
  }, Q = function (z) {
    return it ? null : T(z);
  }, K = function (z) {
    ht && z.stopPropagation();
  }, U = P.useMemo(function () {
    return function () {
      var z = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, V = z.refKey, et = V === void 0 ? "ref" : V, lt = z.role, at = z.onKeyDown, ft = z.onFocus, ot = z.onBlur, ct = z.onClick, st = z.onDragEnter, yt = z.onDragOver, gt = z.onDragLeave, Tt = z.onDrop, kt = Ro(z, lb);
      return Jt(Jt(Xl({
        onKeyDown: R(Cn(at, N)),
        onFocus: R(Cn(ft, W)),
        onBlur: R(Cn(ot, X)),
        onClick: T(Cn(ct, I)),
        onDragEnter: Q(Cn(st, S)),
        onDragOver: Q(Cn(yt, k)),
        onDragLeave: Q(Cn(gt, D)),
        onDrop: Q(Cn(Tt, C)),
        role: typeof lt == "string" && lt !== "" ? lt : "presentation"
      }, et, l), !o && !dt ? {
        tabIndex: 0
      } : {}), kt);
    };
  }, [l, N, W, X, I, S, k, D, C, dt, it, o]), B = P.useCallback(function (z) {
    z.stopPropagation();
  }, []), Z = P.useMemo(function () {
    return function () {
      var z = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, V = z.refKey, et = V === void 0 ? "ref" : V, lt = z.onChange, at = z.onClick, ft = Ro(z, cb), ot = Xl({
        accept: F,
        multiple: _,
        type: "file",
        style: {
          border: 0,
          clip: "rect(0, 0, 0, 0)",
          clipPath: "inset(50%)",
          height: "1px",
          margin: "0 -1px -1px 0",
          overflow: "hidden",
          padding: 0,
          position: "absolute",
          width: "1px",
          whiteSpace: "nowrap"
        },
        onChange: T(Cn(lt, C)),
        onClick: T(Cn(at, B)),
        tabIndex: -1
      }, et, a);
      return Jt(Jt({}, ot), ft);
    };
  }, [a, n, _, C, o]);
  return Jt(Jt({}, i), {}, {
    isFocused: s && !o,
    getRootProps: U,
    getInputProps: Z,
    rootRef: l,
    inputRef: a,
    open: T(O)
  });
}
function yb(t, e) {
  switch (e.type) {
    case "focus":
      return Jt(Jt({}, t), {}, {
        isFocused: !0
      });
    case "blur":
      return Jt(Jt({}, t), {}, {
        isFocused: !1
      });
    case "openDialog":
      return Jt(Jt({}, Yl), {}, {
        isFileDialogActive: !0
      });
    case "closeDialog":
      return Jt(Jt({}, t), {}, {
        isFileDialogActive: !1
      });
    case "setDraggedFiles":
      return Jt(Jt({}, t), {}, {
        isDragActive: e.isDragActive,
        isDragAccept: e.isDragAccept,
        isDragReject: e.isDragReject
      });
    case "setFiles":
      return Jt(Jt({}, t), {}, {
        acceptedFiles: e.acceptedFiles,
        fileRejections: e.fileRejections,
        isDragReject: e.isDragReject
      });
    case "reset":
      return Jt({}, Yl);
    default:
      return t;
  }
}
function tf() {
}
const bb = "data:image/svg+xml,%3Csvg version='1.1' id='Ebene_1' x='0px' y='0px' width='595.275px' height='595.275px' viewBox='200 215 230 470' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3C/defs%3E%3Crect x='125.3' y='264.6' width='350.378' height='349.569' style='fill: rgb(237, 0, 0); stroke: rgb(197, 2, 2);' rx='58.194' ry='58.194'%3E%3C/rect%3E%3Cg%3E%3Crect x='267.162' y='307.978' transform='matrix(0.7071 -0.7071 0.7071 0.7071 -222.6202 340.6915)' style='fill:white;' width='65.545' height='262.18' rx='32.772' ry='32.772'%3E%3C/rect%3E%3Crect x='266.988' y='308.153' transform='matrix(0.7071 0.7071 -0.7071 0.7071 398.3889 -83.3116)' style='fill:white;' width='65.544' height='262.179' rx='32.772' ry='32.772'%3E%3C/rect%3E%3C/g%3E%3C/svg%3E", xb = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7", _b = [{ title: "Load", icon: Gv, disabledInMode: null }, { title: "Add Edge Brush", icon: Kv, disabledInMode: "COLOR" }, { title: "Remove Edge Brush", icon: $v, disabledInMode: "COLOR" }, { title: "Brush", icon: Zv, disabledInMode: "EDGE" }, { title: "Erase", icon: ey, disabledInMode: null }, { title: "Select", icon: iy, disabledInMode: null }], wb = P.forwardRef(({
  theme: t = "system",
  autoAdaptToPhone: e = !0,
  changeDimensionCallBack: n,
  uploadImgCallBack: o,
  uploadOriginalImgCallBack: d,
  uploadAddEdgeMaskCallBack: g,
  uploadRemoveEdgeMaskCallBack: b,
  uploadColoredImgCallBack: _,
  uploadTotalMaskCallBack: w,
  uploadBackgroundImgCallBack: E,
  guessPromptCallBack: j,
  updatePromptCallBack: H
}, G) => {
  P.useEffect(() => {
    if ((t === "dark" || t === "light") && document.body.setAttribute("data-theme", t), t === "system") {
      const $ = window.matchMedia(
        "(prefers-color-scheme: dark)"
      );
      $.matches && document.body.setAttribute("data-theme", "dark"), $.addEventListener("change", (bt) => document.body.setAttribute("data-theme", bt.matches ? "dark" : "light"));
    }
  }, [t]);
  const J = P.useRef(0), nt = P.useRef(null), rt = P.useRef(null), vt = P.useRef(null), Y = P.useRef(null), L = P.useRef(null), A = P.useRef(null), tt = P.useRef(null), dt = P.useRef(null), it = P.useRef(null), ht = P.useRef(null), [v, Ct] = P.useState(!1), [F, Et] = P.useState("Brush"), Ht = P.useRef(F), [It, l] = P.useState({
    Brush: 5,
    Erase: 5,
    "Add Edge Brush": 5,
    "Remove Edge Brush": 5
  }), a = P.useRef({
    add_edge: !1,
    remove_edge: !1,
    color: !1
  }), [c, f] = P.useState(150), [i, r] = P.useState(5), [s, u] = P.useState("#FFFFFF"), [h, p] = P.useState(1), [m, y] = P.useState(""), [x, S] = P.useState(!0), k = P.useRef(x);
  P.useEffect(() => {
    k.current = x;
  }, [x]);
  const [D, M] = P.useState(0), C = P.useRef(D);
  P.useEffect(() => {
    C.current = D;
  }, [D]);
  const [O, N] = P.useState(!1), W = P.useRef(O);
  P.useEffect(() => {
    W.current = O;
  }, [O]);
  const [X, I] = P.useState([]), T = P.useRef([]), R = P.useRef([]), [Q, K] = P.useState(!1), [U, B] = P.useState(!1), [Z, z] = P.useState(!1);
  P.useEffect(() => {
    console.log(F), !(!A.current || !A.current.lowerCanvasEl) && (F === "Select" || F === "Load" || (Ht.current = F, r(It[F]), A.current.freeDrawingBrush.width = It[F], F === "Brush" && (A.current.freeDrawingBrush.color = Hh(
      s,
      h
    )), lt()));
  }, [F]);
  const V = ($, bt) => {
    var Ot = bt.target, Pt = Ot.canvas;
    Ot._objects ? (Pt.remove(...Ot._objects), Pt.discardActiveObject()) : Pt.remove(Ot), Pt.requestRenderAll(), Ge(), Re(), yt();
  }, et = ($) => {
    const bt = Math.min(A.current.width, A.current.height), Ot = document.createElement("img");
    Ot.src = bb, $.hasOwnProperty("controls") || ($.cornerSize = bt / 30, $.controls.removeControl = new Dn.Control({
      x: 0.5,
      y: -0.5,
      offsetY: -16,
      offsetX: 16,
      cursorStyle: "pointer",
      mouseUpHandler: V.bind(void 0),
      render: Yv(Ot),
      cornerSize: bt / 25
    }));
  }, lt = () => {
    let $ = A.current.freeDrawingBrush.width;
    A.current.lowerCanvasEl.clientWidth && ($ = $ * A.current.lowerCanvasEl.clientWidth / A.current.getWidth());
    const bt = A.current.freeDrawingBrush.color, Ot = Ht.current;
    let Pt;
    Ot === "Brush" ? Pt = `
      <svg
        height="${$}"
        viewBox="0 0 ${$ * 2} ${$ * 2}"
        width="${$}"
        xmlns="http://www.w3.org/2000/svg"
      >
      <circle
        cx="50%"
        cy="50%"
        r="${$}"
        fill="${bt}"
        opacity="0.7"
      />
      </svg>
    ` : Pt = `
      <svg
        height="${$}"
        viewBox="0 0 ${$ * 2} ${$ * 2}"
        width="${$}"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle
          cx="50%"
          cy="50%"
          r="${$}"
          fill="#ffffff"
          opacity="0.7"
        />
        <circle
          cx="50%"
          cy="50%"
          r="${$ * 0.9}"
          fill="#000000"
          opacity="0.7"
        />
      </svg>
    `;
    const Mt = `data:image/svg+xml;base64,${window.btoa(Pt)}`;
    A.current.freeDrawingCursor = `url(${Mt}) ${$ / 2} ${$ / 2}, crosshair`;
  }, at = ($, bt) => {
    const Ot = nt.current.clientWidth <= 600 && e;
    tt.current.style.flexDirection = Ot ? "column" : "row", console.log(nt.current.style), (!$ || !bt) && ($ = tt.current.clientWidth, Ot || ($ /= 2), bt = $), A.current.setWidth($), A.current.setHeight(bt);
    let Pt;
    Ot ? Pt = tt.current.clientWidth : Pt = tt.current.clientWidth / 2;
    const Mt = bt / $ * Pt;
    L.current.children[0].style.width = Pt + "px", L.current.children[0].style.height = Mt + "px", L.current.children[0].children[0].style.width = Pt + "px", L.current.children[0].children[0].style.height = Mt + "px", L.current.children[0].children[1].style.width = Pt + "px", L.current.children[0].children[1].style.height = Mt + "px", dt.current.style.width = Pt + "px", dt.current.style.height = Mt + "px", vt.current.style.height = Ot ? Mt * 2 : Mt + "px";
    const Zt = Mt + rt.current.clientHeight, wt = rt.current.clientWidth;
    n({ height: Zt, width: wt }), ft();
  }, ft = () => {
    const $ = Math.min(A.current.width, A.current.height);
    if ($ === 0)
      return;
    const bt = Math.floor($ * 0.1), Ot = Math.floor(bt / 5) * 5, Pt = Math.max(25, Math.min(Ot, 150));
    f(Pt);
    const Mt = Math.floor(Ot / 5 / 3) * 5;
    r(Mt), Object.keys(It).forEach((Zt) => It[Zt] = Mt), l(It);
  }, ot = async ($) => {
    const bt = await new Promise((Ot) => Dn.Image.fromURL($, Ot));
    A.current.clear(), Li(null), gt(), mr(), bt.set({ erasable: !1 }), A.current.setBackgroundImage(
      bt,
      () => {
        A.current.renderAll(), Ge(!0);
      }
    ), Re(), at(bt.width, bt.height), yt();
  }, ct = () => {
    let $ = !W.current;
    A.current.discardActiveObject(), A.current.getObjects().forEach((bt) => {
      bt.set("visible", !$);
    }), N($), A.current.renderAll();
  }, st = () => {
    T.current.length > 1 ? K(!0) : K(!1), R.current.length > 0 ? B(!0) : B(!1);
  }, yt = () => {
    const $ = A.current.toJSON(["brushType"]);
    T.current.push($), R.current = [], st();
  }, gt = () => {
    T.current = [], R.current = [], st();
  }, Tt = ($) => {
    console.log($);
    let bt = $, Ot = new FileReader();
    Ot.onload = (Pt) => {
      z(!1);
      let Mt = Pt.target.result;
      kt(Mt);
    }, Ot.readAsDataURL(bt);
  }, kt = async ($) => {
    di(!0);
    const bt = await E($);
    bt && ot(bt), di(!1);
  }, jt = () => {
    if (T.current.length > 1) {
      const $ = T.current.pop();
      R.current.push($);
      const bt = T.current[T.current.length - 1];
      A.current.loadFromJSON(bt, () => {
        A.current.renderAll(), Ge(), Re();
      });
    }
    st();
  }, Kt = () => {
    if (R.current.length > 0) {
      const $ = R.current.pop();
      T.current.push($), A.current.loadFromJSON($, () => {
        A.current.renderAll(), Ge(), Re();
      });
    }
    st();
  }, Rt = () => new Promise(
    ($) => A.current.clone($, ["enableRetinaScaling", "brushType"])
  ), Bt = async () => {
    const $ = await Rt();
    return $.setHeight(A.current.height), $.setWidth(A.current.width), $.renderAll(), new Promise((bt) => $.lowerCanvasEl.toBlob(bt, "image/png"));
  }, se = async () => {
    const $ = await Rt();
    return $.setHeight(A.current.height), $.setWidth(A.current.width), $.remove(...$.getObjects()), $.renderAll(), new Promise((bt) => $.lowerCanvasEl.toBlob(bt, "image/png"));
  }, zt = async ($ = "all") => {
    const bt = await Rt();
    return bt.setHeight(A.current.height), bt.setWidth(A.current.width), bt.getObjects().forEach((Ot) => {
      Ot.type === "path" && ($ === "Add Edge Brush" || $ === "Remove Edge Brush") && Ot.set("visible", Ot.brushType === $), Ot.fill && Ot.set("fill", "rgb(0,0,0)"), Ot.stroke && Ot.set("stroke", "rgb(0,0,0)"), Ot.set("globalCompositeOperation", "destination-out");
    }), bt.setBackgroundImage(null), bt.setBackgroundColor("black"), bt.renderAll(), new Promise((Ot) => bt.lowerCanvasEl.toBlob(Ot, "image/png"));
  }, ie = async () => {
    const $ = await Rt();
    return $.setHeight(A.current.height), $.setWidth(A.current.width), $.remove(...$.getObjects().filter((bt) => ["Add Edge Brush", "Remove Edge Brush"].includes(bt.brushType))), $.renderAll(), new Promise((bt) => $.lowerCanvasEl.toBlob(bt, "image/png"));
  }, fi = () => {
    const $ = A.current.getObjects().filter((Zt) => Zt.type === "path" && Zt.brushType === "Add Edge Brush"), bt = A.current.getObjects().filter((Zt) => Zt.type === "path" && Zt.brushType === "Remove Edge Brush"), Ot = A.current.getObjects().filter((Zt) => Zt.type === "path" && Zt.brushType !== "Add Edge Brush" && Zt.brushType !== "Remove Edge Brush"), Pt = {
      add_edge: $.length > 0,
      remove_edge: bt.length > 0,
      color: Ot.length > 0
    }, Mt = {
      add_edge: a.current.add_edge || Pt.add_edge,
      remove_edge: a.current.remove_edge || Pt.remove_edge,
      color: a.current.color || Pt.color
    };
    return a.current = Mt, Mt;
  }, mr = () => {
    y(""), H("");
  }, vr = ($) => {
    y($), H($);
  }, ma = P.useCallback(async () => {
    if (k.current)
      try {
        ht.current.classList.remove("shining"), ht.current.classList.add("blinking");
        const $ = await j();
        ht.current.classList.remove("blinking"), !$.error && $.prompt && $.prompt !== "" ? $.prompt != m && (ht.current.classList.add("shining"), setTimeout(() => ht.current.classList.remove("shining"), 1500), y($.prompt)) : mr();
      } catch ($) {
        console.log($);
      }
  }, [x]), di = ($) => {
    $ === !0 ? C.current = C.current + 1 : C.current = Math.max(0, C.current - 1), M(C.current);
  }, [pi, Li] = P.useState(null);
  P.useImperativeHandle(G, () => ({
    updateGeneratedImg($) {
      Li("data:image/png;base64," + $);
    }
  }));
  const va = () => {
    ot(pi);
  }, { getRootProps: ns, getInputProps: Ve } = c0({
    onDrop: ($) => {
      !$ || $.length < 1 || Tt($[0]);
    },
    accept: { "image/*": [] }
  }), Ge = async ($ = !1) => {
    W.current && ct(), di(!0);
    try {
      const bt = await Bt(), Ot = await o(bt);
      if ($) {
        const Zt = await se();
        await d(Zt);
      }
      const Pt = fi();
      if (Pt.add_edge || $) {
        const Zt = await zt("Add Edge Brush");
        await g(Zt);
      }
      if (Pt.remove_edge || $) {
        const Zt = await zt("Remove Edge Brush");
        await b(Zt);
      }
      if (Pt.color || $) {
        const Zt = await ie();
        await _(Zt);
      }
      (Pt.add_edge || Pt.color) && !$ && (A.current.getObjects().length === 0 ? y("") : await ma());
      const Mt = await zt();
      await w(Mt);
    } catch (bt) {
      console.error("Error in uploadPaintFile:", bt), alert("An error occurred while uploading the file.");
    } finally {
      di(!1);
    }
  }, ya = () => {
    A.current.discardActiveObject(), A.current.remove(...A.current.getObjects()), A.current.renderAll(), Ge(), yt(), Re();
  }, yr = ($) => {
    if (A.current.discardActiveObject(), $ !== "Load" && Et($), $.includes("Brush")) {
      if ($ === "Brush")
        A.current.freeDrawingBrush = new Dn.PencilBrush(A.current);
      else if ($ === "Add Edge Brush" || $ === "Remove Edge Brush") {
        const bt = $ === "Add Edge Brush", Ot = new Dn.PatternBrush(A.current);
        Ot.getPatternSrc = function () {
          const Pt = Dn.document.createElement("canvas");
          Pt.width = Pt.height = 15;
          const Mt = Pt.getContext("2d");
          return Mt.fillStyle = bt ? "white" : "black", Mt.fillRect(0, 0, 15, 15), new Dn.Rect({
            width: 10,
            height: 10 / 3,
            fill: bt ? "black" : "white",
            left: 5 / 2,
            top: (15 - 10 / 3) / 2
          }).render(Mt), bt && new Dn.Rect({
            width: 3.3333333333333335,
            height: 10,
            fill: "black",
            left: 5.833333333333333,
            top: 2.5
          }).render(Mt), Pt;
        }, Ot.source = Ot.getPatternSrc(), A.current.freeDrawingBrush = Ot;
      }
      A.current.isDrawingMode = !0;
    } else $ === "Erase" ? (A.current.freeDrawingBrush = new Dn.EraserBrush(A.current), A.current.isDrawingMode = !0) : $ === "Select" ? A.current.isDrawingMode = !1 : $ === "Load" && it.current.click();
  }, Re = () => {
    let $ = [], bt = !1, Ot = !1;
    const Pt = A.current.getObjects().map((Mt, Zt) => {
      const wt = Mt.type;
      let q;
      wt === "path" && (Mt.brushType === "Add Edge Brush" ? (q = "+ edge", bt = !0) : Mt.brushType === "Remove Edge Brush" ? (q = "- edge", bt = !0) : (q = "color", Ot = !0));
      let mt = $.filter((Wt) => Wt === q).length + 1;
      const _t = q + ` ${mt}`;
      return $.unshift(q), {
        name: _t,
        setActiveFunc: () => {
          A.current.discardActiveObject(), et(Mt), A.current.setActiveObject(Mt), A.current.renderAll();
        },
        removeFunc: (Wt) => {
          V(null, { target: Mt }), Wt.stopPropagation();
        },
        isActive: A.current.getActiveObjects().includes(Mt)
      };
    });
    Ct(bt ? "EDGE" : Ot ? "COLOR" : "NONE"), I(Pt.reverse());
  };
  P.useEffect(() => {
    if (tt.current && Y.current) {
      const Ot = new Dn.Canvas(Y.current, {
        isDrawingMode: !0,
        backgroundColor: "transparent",
        enablePointerEvents: !0,
        enableRetinaScaling: !1,
        fireRightClick: !0,
        fireMiddleClick: !0,
        stopContextMenu: !0
      });
      Ot.backgroundColor = "#000000", Ot.on({
        "selection:created": (Pt) => {
          Et("Select"), A.current.isDrawingMode = !1, Re();
        },
        "selection:updated": (Pt) => {
          Re();
        },
        "selection:cleared": (Pt) => {
          Et("Select"), Re();
        },
        "object:modified": (Pt) => {
          Ge(), yt(), Re();
        },
        "mouse:up": (Pt) => {
          A.current.getActiveObjects().forEach((Mt) => {
            et(Mt);
          }), A.current.isDrawingMode && (Ge(), yt());
        },
        "path:created": ({ path: Pt }) => {
          Pt.brushType = Ht.current, Ht.current === "Remove Edge Brush" && A.current.getObjects().forEach((Mt) => {
            Mt.type === "path" && Mt.brushType === "Remove Edge Brush" || Pt.sendBackwards();
          }), Re();
        }
      }), A.current = Ot, at(null, null), z(!0), yr("Add Edge Brush"), yt();
    }
    const $ = (Ot) => {
      for (let Pt of Ot) {
        let Mt = Pt.contentRect.width;
        Mt !== J.current && (J.current = Mt, !A.current.width || !A.current.height ? at(Mt, Mt) : at(A.current.width, A.current.height));
      }
    }, bt = new ResizeObserver($);
    return nt.current && bt.observe(nt.current), () => {
      nt.current && bt.unobserve(nt.current), A.current && A.current.dispose();
    };
  }, []), P.useEffect(() => {
    A.current && (A.current.freeDrawingBrush.color = Hh(
      s,
      h
    ), lt());
  }, [s, h]), P.useEffect(() => {
    A.current && (A.current.freeDrawingBrush.width = Number(i), It[F] = Number(i), l(It), lt());
  }, [i]);
  const ba = () => {
    let $ = document.createElement("a");
    pi ? $.href = pi : $.href = A.current.toDataURL({
      format: "png",
      quality: 0.8
    }), $.download = "magicquill.png", $.click();
  };
  return /* @__PURE__ */ St.jsxs("div", {
    style: { height: "100%", width: "100%" }, ref: nt, children: [
    /* @__PURE__ */ St.jsxs("div", {
      className: "top-bar", ref: rt, children: [
      /* @__PURE__ */ St.jsxs("div", {
        id: "prompt-input-box", children: [
        /* @__PURE__ */ St.jsx("div", {
          className: "painter-tool-icon", children: /* @__PURE__ */ St.jsx(
            ay,
            {
              id: "wand-icon",
              ref: ht,
              className: x ? "active" : "",
              title: "Guess prompt: " + (x ? "on" : "off"),
              onClick: () => S(!x)
            }
          )
        }),
        /* @__PURE__ */ St.jsx("input", { id: "prompt-input", placeholder: "Brush your canvas to tell us what you'd like to create!", onChange: ($) => vr($.target.value), value: m })
        ]
      }),
      /* @__PURE__ */ St.jsxs("div", {
        id: "tool-bar", children: [
        /* @__PURE__ */ St.jsxs("div", {
          id: "essential-tool-box", children: [
          /* @__PURE__ */ St.jsx("input", { type: "file", accept: "image/jpeg,image/png,image/webp", hidden: !0, ref: it, onChange: ($) => Tt($.target.files[0]) }),
            _b.map(($) => {
              const bt = $.icon;
              return /* @__PURE__ */ St.jsx(
                "div",
                {
                  className: "painter-tool-icon" + (v === $.disabledInMode ? " disabled" : ""),
                  title: $.title,
                  onClick: () => {
                    $.disabledInMode !== v && yr($.title);
                  },
                  children: /* @__PURE__ */ St.jsx(bt, { className: F === $.title ? "active" : "" })
                },
                $.title
              );
            })
          ]
        }),
        /* @__PURE__ */ St.jsxs("div", {
          id: "stroke-size-box", children: [
          /* @__PURE__ */ St.jsx("div", { className: "separator" }),
          /* @__PURE__ */ St.jsx("input", { type: "range", id: "stroke-width-slider", min: "1", max: c, value: i, step: "1", title: "Stroke width", onChange: ($) => r($.target.value) }),
          /* @__PURE__ */ St.jsx("div", { id: "stroke-width-value", children: i })
          ]
        }),
        /* @__PURE__ */ St.jsxs("div", {
          id: "color-and-alpha-box", children: [
          /* @__PURE__ */ St.jsx("div", { className: "separator" }),
            F === "Brush" && /* @__PURE__ */ St.jsxs(St.Fragment, {
              children: [
            /* @__PURE__ */ St.jsx("input", { id: "stroke-color", type: "color", value: s, title: "Stroke color", onChange: ($) => u($.target.value) }),
            /* @__PURE__ */ St.jsx("input", { id: "stroke-color-transparent", type: "number", max: "1.0", min: "0.1", step: "0.05", value: h, title: "Stroke alpha value", onChange: ($) => p($.target.value) })
              ]
            }),
          /* @__PURE__ */ St.jsxs("div", {
              id: "painter-history-panel", children: [
            /* @__PURE__ */ St.jsx("div", { className: "separator" }),
            /* @__PURE__ */ St.jsx("div", { className: "painter-tool-icon" + (Q ? "" : " disabled"), title: "Undo", children: /* @__PURE__ */ St.jsx(sy, { title: "Undo", onClick: () => jt() }) }),
            /* @__PURE__ */ St.jsx("div", { className: "painter-tool-icon" + (U ? "" : " disabled"), title: "Redo", children: /* @__PURE__ */ St.jsx(cy, { title: "Redo", onClick: () => Kt() }) })
              ]
            })
          ]
        }),
        /* @__PURE__ */ St.jsx("div", { id: "download-box", children: /* @__PURE__ */ St.jsx("div", { className: "painter-tool-icon", children: /* @__PURE__ */ St.jsx(xy, { title: "Download", onClick: () => ba() }) }) })
        ]
      })
      ]
    }),
    /* @__PURE__ */ St.jsxs("div", {
      className: "lower-area", ref: vt, children: [
      /* @__PURE__ */ St.jsxs("div", {
        className: "side-bar", children: [
        /* @__PURE__ */ St.jsxs("div", {
          className: "layer-box", children: [
          /* @__PURE__ */ St.jsxs("div", {
            className: "layer-operation-box", children: [
            /* @__PURE__ */ St.jsx("div", { className: "painter-tool-icon", title: "Remove All", id: "layer-remove-button", children: /* @__PURE__ */ St.jsx(Wh, { title: "Remove All", onClick: () => ya() }) }),
            /* @__PURE__ */ St.jsx("div", { className: "painter-tool-icon", title: "Hide All", id: "layer-hide-button", onClick: () => ct(), children: /* @__PURE__ */ St.jsx(my, { className: O ? "active" : "", title: "Hide All" }) })
            ]
          }),
          /* @__PURE__ */ St.jsx("div", {
            className: "layer-item-list", children: X.map(($) => /* @__PURE__ */ St.jsxs("div", {
              className: "layer-item-container" + ($.isActive ? " active" : ""), onClick: () => $.setActiveFunc(), children: [
            /* @__PURE__ */ St.jsx("div", { className: "layer-item", children: $.name }),
            /* @__PURE__ */ St.jsx(Wh, { className: "layer-item-remove", title: "Remove", onClick: (bt) => $.removeFunc(bt) })
              ]
            }, $.name))
          })
          ]
        }),
        /* @__PURE__ */ St.jsx("div", { id: "loading-status", hidden: D === 0, children: /* @__PURE__ */ St.jsx(yy, { className: "rotating" }) })
        ]
      }),
      /* @__PURE__ */ St.jsxs("div", {
        id: "output-area", ref: tt, children: [
        /* @__PURE__ */ St.jsxs("div", {
          id: "left-box", children: [
            Z && /* @__PURE__ */ St.jsxs("div", {
              ...ns({ id: "bg-reminder" }), children: [
            /* @__PURE__ */ St.jsx("input", { ...Ve() }),
            /* @__PURE__ */ St.jsx("p", { id: "bg-reminder-text", children: "Upload or drop an image here" })
              ]
            }),
          /* @__PURE__ */ St.jsx("div", { id: "canvas-box", ref: L, children: /* @__PURE__ */ St.jsx("canvas", { ref: Y }) })
          ]
        }),
        /* @__PURE__ */ St.jsxs("div", {
          id: "output-img-box", ref: dt, children: [
            pi && /* @__PURE__ */ St.jsxs("div", {
              id: "output-operation-box", children: [
            /* @__PURE__ */ St.jsx("div", { className: "painter-tool-icon", title: "Accept", onClick: () => va(), children: /* @__PURE__ */ St.jsx(hy, { title: "Accept", style: { fill: "#507B58" } }) }),
            /* @__PURE__ */ St.jsx("div", { className: "painter-tool-icon", title: "Discard", onClick: () => Li(null), children: /* @__PURE__ */ St.jsx(dy, { title: "Discard", style: { fill: "#AB3131" } }) })
              ]
            }),
          /* @__PURE__ */ St.jsx("img", { id: "output-img", src: pi || xb })
          ]
        })
        ]
      })
      ]
    })
    ]
  });
}), Cb = (t, e) => {
  console.log(t, e, "embedMagicQuill");
  const n = Math.random().toString(36).substring(7);
  if (!t)
    throw "DOM element not found.";
  const o = Ff.createRef();
  return window.updateGeneratedImg || (window.updateGeneratedImg = {}), window.updateGeneratedImg[n] = (d) => {
    o.current && o.current.updateGeneratedImg(d);
  }, kv.render(
    // <React.StrictMode>
    /* @__PURE__ */ St.jsx(wb, { ...e, ref: o }),
    // </React.StrictMode>,
    t
  ), n;
}, {
  SvelteComponent: Sb,
  assign: Eb,
  binding_callbacks: kb,
  check_outros: Tb,
  create_component: u0,
  destroy_component: h0,
  detach: ef,
  element: Ob,
  flush: Va,
  get_spread_object: Pb,
  get_spread_update: Mb,
  group_outros: Ab,
  init: Db,
  insert: nf,
  mount_component: f0,
  safe_not_equal: jb,
  space: Lb,
  transition_in: Vr,
  transition_out: Bo
} = window.__gradio__svelte__internal, { onMount: Fb } = window.__gradio__svelte__internal;
function rf(t) {
  let e, n;
  const o = [
    { i18n: "" },
    { autoscroll: !0 },
    { translucent: !0 },
    /*loading_status*/
    t[0]
  ];
  let d = {};
  for (let g = 0; g < o.length; g += 1)
    d = Eb(d, o[g]);
  return e = new Lg({ props: d }), {
    c() {
      u0(e.$$.fragment);
    },
    m(g, b) {
      f0(e, g, b), n = !0;
    },
    p(g, b) {
      const _ = b & /*loading_status*/
        1 ? Mb(o, [
          o[0],
          o[1],
          o[2],
          Pb(
            /*loading_status*/
            g[0]
          )
        ]) : {};
      e.$set(_);
    },
    i(g) {
      n || (Vr(e.$$.fragment, g), n = !0);
    },
    o(g) {
      Bo(e.$$.fragment, g), n = !1;
    },
    d(g) {
      h0(e, g);
    }
  };
}
function Ib(t) {
  let e, n, o, d = (
    /*loading_status*/
    t[0] && rf(t)
  );
  return {
    c() {
      d && d.c(), e = Lb(), n = Ob("div");
    },
    m(g, b) {
      d && d.m(g, b), nf(g, e, b), nf(g, n, b), t[5](n), o = !0;
    },
    p(g, b) {
      /*loading_status*/
      g[0] ? d ? (d.p(g, b), b & /*loading_status*/
        1 && Vr(d, 1)) : (d = rf(g), d.c(), Vr(d, 1), d.m(e.parentNode, e)) : d && (Ab(), Bo(d, 1, 1, () => {
          d = null;
        }), Tb());
    },
    i(g) {
      o || (Vr(d), o = !0);
    },
    o(g) {
      Bo(d), o = !1;
    },
    d(g) {
      g && (ef(e), ef(n)), d && d.d(g), t[5](null);
    }
  };
}
function Rb(t) {
  let e, n;
  return e = new O0({
    props: {
      $$slots: { default: [Ib] },
      $$scope: { ctx: t }
    }
  }), {
    c() {
      u0(e.$$.fragment);
    },
    m(o, d) {
      f0(e, o, d), n = !0;
    },
    p(o, [d]) {
      const g = {};
      d & /*$$scope, el, loading_status*/
        2097155 && (g.$$scope = { dirty: d, ctx: o }), e.$set(g);
    },
    i(o) {
      n || (Vr(e.$$.fragment, o), n = !0);
    },
    o(o) {
      Bo(e.$$.fragment, o), n = !1;
    },
    d(o) {
      h0(e, o);
    }
  };
}
function Bb(t, e, n) {
  var o = this && this.__awaiter || function (v, Ct, F, Et) {
    function Ht(It) {
      return It instanceof F ? It : new F(function (l) {
        l(It);
      });
    }
    return new (F || (F = Promise))(function (It, l) {
      function a(i) {
        try {
          f(Et.next(i));
        } catch (r) {
          l(r);
        }
      }
      function c(i) {
        try {
          f(Et.throw(i));
        } catch (r) {
          l(r);
        }
      }
      function f(i) {
        i.done ? It(i.value) : Ht(i.value).then(a, c);
      }
      f((Et = Et.apply(v, Ct || [])).next());
    });
  };
  let { value: d = {
    from_frontend: {
      img: null,
      original_image: null,
      add_edge_image: null,
      remove_edge_image: null,
      add_color_image: null,
      total_mask: null
    },
    from_backend: { prompt: null, generated_image: null }
  } } = e, { loading_status: g = void 0 } = e, b = null, { theme: _ = "system" } = e, { url: w = void 0 } = e, E, j;
  function H() {
    if (d.from_backend !== void 0 && d.from_backend.generated_image !== null && d.from_backend.generated_image !== b && window.updateGeneratedImg && window.updateGeneratedImg[j]) {
      const v = window.updateGeneratedImg[j];
      v(d.from_backend.generated_image), b = d.from_backend.generated_image;
    }
  }
  const G = (v) => new Promise((Ct, F) => {
    const Et = new FileReader();
    Et.onloadend = () => Ct(Et.result), Et.readAsDataURL(v);
  }), J = (v) => console.log(v), nt = (v) => o(void 0, void 0, void 0, function* () {
    const Ct = yield G(v);
    n(2, d.from_frontend.img = Ct, d);
  }), rt = (v) => o(void 0, void 0, void 0, function* () {
    const Ct = yield G(v);
    n(2, d.from_frontend.original_image = Ct, d);
  }), vt = (v) => o(void 0, void 0, void 0, function* () {
    const Ct = yield G(v);
    n(2, d.from_frontend.add_edge_image = Ct, d);
  }), Y = (v) => o(void 0, void 0, void 0, function* () {
    const Ct = yield G(v);
    n(2, d.from_frontend.remove_edge_image = Ct, d);
  }), L = (v) => o(void 0, void 0, void 0, function* () {
    const Ct = yield G(v);
    n(2, d.from_frontend.add_color_image = Ct, d);
  }), A = (v) => o(void 0, void 0, void 0, function* () {
    const Ct = yield G(v);
    n(2, d.from_frontend.total_mask = Ct, d);
  }), tt = (v) => o(void 0, void 0, void 0, function* () {
    return yield fetch(`${w || ""}/magic_quill/process_background_img`, {
      method: "POST",
      body: JSON.stringify(v),
      headers: { "content-type": "application/json" }
    }).then((F) => F.json());
  }), dt = (v) => o(void 0, void 0, void 0, function* () {
    const Ct = yield fetch(`${w || ""}/magic_quill/guess_prompt`, {
      method: "POST",
      body: JSON.stringify(d.from_frontend),
      headers: { "content-type": "application/json" }
    }).then((F) => F.json());
    return n(2, d.from_backend.prompt = Ct, d), { error: !1, prompt: Ct };
  }), it = (v) => {
    n(2, d.from_backend.prompt = v, d);
  };
  Fb(() => {
    j = Cb(E, {
      theme: _,
      changeDimensionCallBack: J,
      uploadImgCallBack: nt,
      uploadOriginalImgCallBack: rt,
      uploadAddEdgeMaskCallBack: vt,
      uploadRemoveEdgeMaskCallBack: Y,
      uploadColoredImgCallBack: L,
      uploadTotalMaskCallBack: A,
      uploadBackgroundImgCallBack: tt,
      guessPromptCallBack: dt,
      updatePromptCallBack: it
    });
  });
  function ht(v) {
    kb[v ? "unshift" : "push"](() => {
      E = v, n(1, E);
    });
  }
  return t.$$set = (v) => {
    "value" in v && n(2, d = v.value), "loading_status" in v && n(0, g = v.loading_status), "theme" in v && n(3, _ = v.theme), "url" in v && n(4, w = v.url);
  }, t.$$.update = () => {
    t.$$.dirty & /*value*/
      4 && d === null && n(2, d = {
        from_frontend: {
          img: null,
          original_image: null,
          add_edge_image: null,
          remove_edge_image: null,
          add_color_image: null,
          total_mask: null
        },
        from_backend: { prompt: null, generated_image: null }
      }), t.$$.dirty & /*value*/
      4 && H();
  }, [g, E, d, _, w, ht];
}
class zb extends Sb {
  constructor(e) {
    super(), Db(this, e, Bb, Rb, jb, {
      value: 2,
      loading_status: 0,
      theme: 3,
      url: 4
    });
  }
  get value() {
    return this.$$.ctx[2];
  }
  set value(e) {
    this.$$set({ value: e }), Va();
  }
  get loading_status() {
    return this.$$.ctx[0];
  }
  set loading_status(e) {
    this.$$set({ loading_status: e }), Va();
  }
  get theme() {
    return this.$$.ctx[3];
  }
  set theme(e) {
    this.$$set({ theme: e }), Va();
  }
  get url() {
    return this.$$.ctx[4];
  }
  set url(e) {
    this.$$set({ url: e }), Va();
  }
}
export {
  zb as default
};
