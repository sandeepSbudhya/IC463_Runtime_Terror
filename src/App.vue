<template>
  <v-app>
    <Toolbar @toggle-drawer="$refs.drawer.drawer = !$refs.drawer.drawer" />
    <router-view></router-view>
    <NavigationDrawer ref="drawer" />
  </v-app>
</template>

<script>
import Toolbar from "./components/home/Toolbar";
import NavigationDrawer from "@/components/home/NavigationDrawer";
export default {
  name: "App",

  components: {
    Toolbar,
    NavigationDrawer
  },

  data: () => ({
    //
  })
};
</script>
