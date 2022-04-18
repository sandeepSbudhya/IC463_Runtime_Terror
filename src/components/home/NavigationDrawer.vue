<template>
  <v-navigation-drawer color="primary" dark app v-if="this.$root.authenticated" v-model="drawer">
    <v-list>
      <v-list-item two-line class="px-3">
        <v-list-item-avatar>
          <img
            src="https://b7.pngbarn.com/png/629/1012/hospital-medicine-health-care-hospital-medicine-health-png-clip-art-thumbnail.png"
          />
        </v-list-item-avatar>
        <v-list-item-content>
          <v-list-item-title>Welcome</v-list-item-title>
          <v-list-item-subtitle>{{this.$root.username}}</v-list-item-subtitle>
        </v-list-item-content>
      </v-list-item>

      <v-list-item v-for="item in items" :key="item.title" router :to="item.route">
        <v-list-item-icon>
          <v-icon>{{ item.icon }}</v-icon>
        </v-list-item-icon>

        <v-list-item-content>
          <v-list-item-title>{{ item.title }}</v-list-item-title>
        </v-list-item-content>
      </v-list-item>
    </v-list>
  </v-navigation-drawer>
</template>

<script>
export default {
  name: "NavigationDrawer",

  components: {},

  data: () => ({
    drawer: true,
    items: [
      { title: "Dashboard", icon: "mdi-view-dashboard", route: "/dash" },
      { title: "History", icon: "mdi-calendar", route: "/history" },
      {
        title: "Settings",
        icon: "mdi-account-settings",
        route: "/settings",
      },
    ],
  }),
};
</script>
